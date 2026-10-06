"""Build the BC + Washington hourly PM2.5 dataset from the AirNow archive (dataset_num 9).

A KnowAir-style benchmark from public reference monitors: hourly PM2.5 from
the British Columbia Ministry of Environment, Metro Vancouver, Environment
and Climate Change Canada (CAPMoN) and the Washington Department of Ecology /
Spokane Regional Clean Air Agency networks, all as reported to AirNow, plus
ERA5 weather at each station. Default window 2024-10-01 .. 2026-09-30.

Sources (all free, no API key):
  - PM2.5: AirNow's public file archive, one file per UTC hour for all of
    North America (files.airnowtech.org/airnow/YYYY/YYYYMMDD/HourlyData_
    YYYYMMDDHH.dat, ~750 kB each, ~13 GB for two years). Only BC/WA PM2.5
    rows are kept, cached per hour, so re-runs and interrupted runs resume.
    Station metadata from the same archive's monitoring_site_locations.dat,
    sampled quarterly. AirNow data are real-time and PRELIMINARY (not the
    agencies' final validated records) - say so in any paper.
  - Weather: ERA5 hourly via the Open-Meteo archive API, at each station's
    0.25-degree ERA5 cell (stations sharing a cell share one request), the
    same 5 variables as datasets 4-8. One location per request, throttled
    under Open-Meteo's free limits (600/min, 5000/h, 10,000/day weighted
    calls; two years at one point weighs ~52), cached per cell.
  - Elevation: Open-Meteo elevation API (AirNow's elevation field is often 0).

Station selection: PM2.5 sites in BC (Canadian sites in 48.2-60N, west of
114W, excluding Alberta/Yukon agencies) and Washington (AQS state code 53),
reporting a valid value (-5..1000 ug/m3, -999 = missing) in >= 85% of hours
across the window. Co-located sites (< 1 km) keep the more complete one.
Values in [-5, 0) are set to 0; gaps are linearly interpolated in time.

Usage: python pilot/build_bcwa_airnow.py [--start 2024-10-01 --end 2026-09-30]
                                         [--cache DIR] [--out DIR] [--workers 8]
Writes <out>/<tag>.npy (float32 [hours, stations, 5 weather + PM2.5]),
<out>/site_<tag lower>.txt (graph.py node file), <out>/<tag>_sites.csv
(names, agencies, completeness) and <out>/<tag>_monthly.csv.
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

AIRNOW = "https://files.airnowtech.org/airnow"
WEATHER_VARS = ["temperature_2m", "relative_humidity_2m", "precipitation",
                "wind_speed_10m", "wind_direction_10m"]
ERA5_DEG = 0.25
OM_PER_MIN, OM_PER_HOUR = 500, 4500     # stay under Open-Meteo's 600/min, 5000/h
DEDUP_KM = 1.0


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def fetch(url, retries=4):
    """Bytes at url; None on 404; retries transient errors."""
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if attempt == retries - 1:
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == retries - 1:
                raise
        time.sleep(5 * (attempt + 1))


# ---------------------------------------------------------------- stations
def bc_border_lon(lat):
    """Approximate BC/Alberta border: 120W north of 53.8N, then roughly the
    continental divide down to 114.05W at 49N."""
    return -120.0 if lat >= 53.8 else -114.05 - (lat - 49.0) / 4.8 * 5.95


def region_of(aqsid, country, lat, lon, agency):
    if (len(aqsid) == 9 and aqsid[:2] == "53") or aqsid[:5] == "84053":
        return "WA"
    if country != "CA" or not 48.2 <= lat < 60.0:
        return None
    # Canadian AirNow IDs embed the NAPS station ID ('000' + NAPS, or
    # '124000' + NAPS), whose first two digits are the province - 10 is BC.
    # Agency names alone let e.g. a federal CAPMoN site in Alberta through.
    naps = aqsid[3:9] if len(aqsid) == 9 else aqsid[6:12] if aqsid[:6] == "124000" else ""
    if naps.isdigit():
        return "BC" if naps[:2] == "10" else None
    return "BC" if lon <= bc_border_lon(lat) and "Alberta" not in agency else None


def site_table(t0, t1, cache):
    fp = os.path.join(cache, "sites_all.csv")
    if os.path.exists(fp):
        return pd.read_csv(fp, dtype={"aqsid": str})
    frames = []
    for d in pd.date_range(t0.normalize(), t1.normalize(), freq="QS").union([t0.normalize(), t1.normalize()]):
        raw = fetch(f"{AIRNOW}/{d:%Y}/{d:%Y%m%d}/monitoring_site_locations.dat")
        if raw is None:
            continue
        rows = [l.split("|") for l in raw.decode("latin-1").splitlines()]
        df = pd.DataFrame([[r[0], r[3], r[6], r[8], r[9], r[12]] for r in rows if len(r) > 12 and r[1] == "PM2.5"],
                          columns=["aqsid", "name", "agency", "lat", "lon", "country"])
        frames.append(df)
    s = pd.concat(frames).drop_duplicates("aqsid", keep="last")
    s[["lat", "lon"]] = s[["lat", "lon"]].astype(float)
    s["region"] = [region_of(*r) for r in zip(s.aqsid, s.country, s.lat, s.lon, s.agency)]
    s = s[s.region.notna()].reset_index(drop=True)
    s.to_csv(fp, index=False)
    return s


# ---------------------------------------------------------------- PM2.5
def hour_file(t, cache):
    return os.path.join(cache, "airnow", f"{t:%Y%m}", f"{t:%Y%m%d%H}.csv")


def get_hour(t, ids, cache):
    """Cache this UTC hour's BC/WA PM2.5 rows; returns 'missing' if AirNow has no file."""
    fp = hour_file(t, cache)
    if os.path.exists(fp):
        return None
    raw = fetch(f"{AIRNOW}/{t:%Y}/{t:%Y%m%d}/HourlyData_{t:%Y%m%d%H}.dat")
    rows = []
    if raw is not None:
        for line in raw.decode("latin-1").splitlines():
            p = line.split("|")
            if len(p) > 7 and p[5] == "PM2.5" and p[2] in ids:
                rows.append(f"{p[2]},{p[7]}")
    os.makedirs(os.path.dirname(fp), exist_ok=True)
    with open(fp + ".tmp", "w") as f:
        f.write("aqsid,pm\n" + "\n".join(rows) + ("\n" if rows else ""))
    os.replace(fp + ".tmp", fp)
    return "missing" if raw is None else None


def pm_table(hours, sites, cache, workers):
    ids = set(sites.aqsid)
    todo = [t for t in hours if not os.path.exists(hour_file(t, cache))]
    print(f"AirNow: {len(hours) - len(todo)} hours cached, {len(todo)} to download")
    missing, t_start = 0, time.time()
    with ThreadPoolExecutor(workers) as ex:
        for i, r in enumerate(ex.map(lambda t: get_hour(t, ids, cache), todo), 1):
            missing += r == "missing"
            if i % 200 == 0 or i == len(todo):
                eta = (time.time() - t_start) / i * (len(todo) - i) / 60
                print(f"\r  {i}/{len(todo)} hours, {missing} without a file, ~{eta:.0f} min left",
                      end="", flush=True)
    print()
    cols = {}
    for t in hours:
        h = pd.read_csv(hour_file(t, cache), dtype={"aqsid": str})
        cols[t] = h.drop_duplicates("aqsid").set_index("aqsid")["pm"]
    pm = pd.DataFrame(cols).T.reindex(columns=sites.aqsid).astype(float)   # [hours, sites]
    pm = pm.where((pm >= -5) & (pm <= 1000))
    return pm.clip(lower=0)


def select_stations(pm, sites, min_comp):
    sites = sites.copy()
    sites["completeness"] = pm.notna().mean().reindex(sites.aqsid).values
    ok = sites[sites.completeness >= min_comp].sort_values("completeness", ascending=False)
    keep = []
    for r in ok.itertuples():
        if all(haversine_km(r.lat, r.lon, k.lat, k.lon) >= DEDUP_KM for k in keep):
            keep.append(r)
    sel = pd.DataFrame(keep).drop(columns="Index")
    print(f"{len(sites)} BC/WA PM2.5 sites; {len(ok)} with >= {min_comp:.0%} of hours; "
          f"{len(sel)} after merging co-located sites -> {sel.region.value_counts().to_dict()}")
    return sel.sort_values(["region", "lat"], ascending=[True, False]).reset_index(drop=True)


# ---------------------------------------------------------------- weather
class OpenMeteo:
    """One-location requests, throttled to Open-Meteo's weighted free limits."""

    def __init__(self, cache):
        self.cache, self.log = cache, []

    def _wait(self, w):
        while True:
            now = time.time()
            self.log = [(t, x) for t, x in self.log if now - t < 3600]
            minute = sum(x for t, x in self.log if now - t < 60)
            if minute + w <= OM_PER_MIN and sum(x for _, x in self.log) + w <= OM_PER_HOUR:
                self.log.append((now, w))
                return
            time.sleep(5)

    def point(self, lat, lon, start, end):
        d = os.path.join(self.cache, "meteo")
        os.makedirs(d, exist_ok=True)
        fp = os.path.join(d, f"{lat:.2f}_{lon:.2f}.json")
        if os.path.exists(fp):
            return json.load(open(fp))
        days = (pd.Timestamp(end) - pd.Timestamp(start)).days + 1
        w = max(1.0, days / 14) * max(1.0, len(WEATHER_VARS) / 10)
        p = dict(latitude=f"{lat:.2f}", longitude=f"{lon:.2f}", start_date=start, end_date=end,
                 hourly=",".join(WEATHER_VARS), wind_speed_unit="ms", models="era5", timezone="GMT")
        url = "https://archive-api.open-meteo.com/v1/archive?" + urllib.parse.urlencode(p)
        for _ in range(30):
            self._wait(w)
            try:
                with urllib.request.urlopen(url, timeout=300) as r:
                    j = json.load(r)
                break
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", "replace")
                if e.code == 429 and "Daily" in body:
                    sys.exit("\nOpen-Meteo daily limit reached - progress is cached; "
                             "re-run this script tomorrow to continue.")
                if e.code in (429, 500, 502, 503, 504):
                    time.sleep(60)
                    continue
                raise
        else:
            raise RuntimeError(f"Open-Meteo kept failing for {lat},{lon}")
        json.dump(j["hourly"], open(fp, "w"))
        return j["hourly"]


def weather_array(sites, hours, cache):
    cell = [(round(a / ERA5_DEG) * ERA5_DEG, round(b / ERA5_DEG) * ERA5_DEG)
            for a, b in zip(sites.lat, sites.lon)]
    cells = sorted(set(cell))
    om = OpenMeteo(cache)
    got = {}
    for i, c in enumerate(cells, 1):
        got[c] = om.point(*c, str(hours[0].date()), str(hours[-1].date()))
        print(f"\rOpen-Meteo: {i}/{len(cells)} ERA5 cells", end="", flush=True)
    print()
    out = []
    for c in cell:
        h = got[c]
        df = pd.DataFrame({v: h[v] for v in WEATHER_VARS},
                          index=pd.to_datetime(h["time"]).tz_localize("UTC"))
        out.append(df.reindex(hours).interpolate().ffill().bfill().to_numpy())
    return np.stack(out, axis=1)                                       # [T, N, 5]


def elevations(sites):
    out = []
    for i in range(0, len(sites), 100):
        s = sites.iloc[i:i + 100]
        p = dict(latitude=",".join(f"{x:.5f}" for x in s.lat), longitude=",".join(f"{x:.5f}" for x in s.lon))
        out += json.loads(fetch("https://api.open-meteo.com/v1/elevation?" + urllib.parse.urlencode(p)))["elevation"]
    return out


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2024-10-01")
    ap.add_argument("--end", default="2026-09-30", help="last day, inclusive")
    ap.add_argument("--tag", default="BCWA_AirNow_2y")
    ap.add_argument("--cache", default=os.path.join(HERE, "cache_airnow"))
    ap.add_argument("--out", default=os.path.join(REPO, "data"))
    ap.add_argument("--min_completeness", type=float, default=0.85)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    t0 = pd.Timestamp(args.start, tz="UTC")
    t1 = pd.Timestamp(args.end, tz="UTC") + pd.Timedelta(hours=23)
    hours = pd.date_range(t0, t1, freq="h")
    os.makedirs(args.cache, exist_ok=True)
    os.makedirs(args.out, exist_ok=True)
    print(f"window {t0} .. {t1} ({len(hours)} h), cache {args.cache}")

    sites = site_table(t0, t1, args.cache)
    print(f"{len(sites)} candidate PM2.5 sites: {sites.region.value_counts().to_dict()}")
    pm_all = pm_table(hours, sites, args.cache, args.workers)
    sel = select_stations(pm_all, sites, args.min_completeness)

    pm = pm_all[sel.aqsid]
    gaps = []
    for c in pm.columns:
        g = pm[c].isna().astype(int)
        gaps.append(int(g.groupby((g == 0).cumsum()).sum().max()))
    sel["longest_gap_h"] = gaps
    print(f"PM2.5: interpolating {pm.isna().to_numpy().mean():.1%} of (hour, station) values; "
          f"longest gap per station: median {np.median(gaps):.0f} h, max {max(gaps)} h")
    pm = pm.interpolate(method="time", limit_direction="both").to_numpy()

    weather = weather_array(sel, hours, args.cache)
    sel["elevation"] = elevations(sel)

    arr = np.concatenate([weather, pm[..., None]], axis=-1).astype(np.float32)
    assert np.isfinite(arr).all()
    np.save(os.path.join(args.out, f"{args.tag}.npy"), arr)
    print(f"wrote {args.tag}.npy {arr.shape}")
    site_fp = os.path.join(args.out, f"site_{args.tag.lower()}.txt")
    with open(site_fp, "w") as f:
        for i, r in enumerate(sel.itertuples()):
            f.write(f"{i} {r.aqsid} {r.lon} {r.lat} {r.elevation}\n")
    sel.to_csv(os.path.join(args.out, f"{args.tag}_sites.csv"), index_label="node")
    print(f"wrote {site_fp}")

    pmf = pd.DataFrame(pm, index=hours)
    month = pmf.index.strftime("%Y-%m")
    monthly = pd.DataFrame({
        "pm_mean": pmf.mean(axis=1).groupby(month).mean(),
        "pm_p99": pmf.groupby(month).apply(lambda g: np.percentile(g.values, 99)),
        "station_hours_gt35.5": (pmf > 35.5).mean(axis=1).groupby(month).mean(),
        "hours_any_gt35.5": (pmf > 35.5).any(axis=1).groupby(month).mean(),
    }).round(3)
    monthly.to_csv(os.path.join(args.out, f"{args.tag}_monthly.csv"))
    print(monthly.to_string())


if __name__ == "__main__":
    main()
