"""Build the multi-year BC wildfire-smoke datasets (dataset_num 8/9/10).

Same recipe as build_bcsmoke_pilot.py (datasets 5/6/7): same node box and
~25 km cells, PurpleAir pm2.5_alt, ERA5 weather via Open-Meteo, NASA FIRMS
VIIRS S-NPP fires, same fire_iso / fire_aniso definitions - stretched from
one 2-month season to a multi-year window (default 2024-10-01 .. 2026-09-30,
the last two full years). What had to change to fit free API tiers:

  - PurpleAir: history pulled in 60-day chunks (the pilot pulled 62 days in
    one hourly call; longer spans aren't documented to work). Candidates in a
    cell are first ranked by a cheap DAILY-average probe, so only the chosen
    sensor pays for the full hourly pull. The sensor list comes from the API
    (feasibility/purpleair_sensors.csv is git-ignored, so a fresh clone - e.g.
    Colab - doesn't have it).
  - Open-Meteo weights a request by locations x days/14, so two years at one
    point costs ~52 of the free 10,000/day, 5,000/hour, 600/minute calls.
    One location per request, throttled under those limits, cached per
    location: if the daily limit stops it, just re-run the next day.
    Fire-source wind comes from a 2-degree grid (the pilot fetched every
    0.25-degree fire cell - ~15x too many calls for two years).
  - FIRMS: 2025+ country archives aren't published yet, so fires come from
    the FIRMS area API (free MAP_KEY, env FIRMS_MAP_KEY): standard-processing
    (SP) data where available, near-real-time (NRT) after. API rows carry no
    'type' column (the pilot's type==0 vegetation filter), so persistent
    non-wildfire hotspots (refineries, flares, mills) are dropped by activity
    outside fire season instead: a 0.25-degree cell with detections on
    >= STATIC_MIN_DAYS distinct Nov-Mar days. Without a MAP_KEY it falls back
    to the yearly country archives, which only works for published years.
  - Fire features are accumulated per fire day from sparse active-cell lists
    (the pilot's dense [hours x cells] arrays don't fit for two years).

PM2.5 gaps are linearly interpolated as in the pilot, so the completeness bar
matters more over two years; the log reports interpolated share and longest
gap. <out>/<tag>_monthly.csv has per-month smoke stats - check it before
trusting the train/val/test split in config.yaml (datasets 8-10).

Usage (local or Colab; re-run to resume, everything is cached):
  PURPLEAIR_API_KEY=... FIRMS_MAP_KEY=... python pilot/build_bcsmoke_long.py
      [--start 2024-10-01 --end 2026-09-30] [--cache DIR] [--out DIR]
Writes <out>/<tag>_{nofire,iso,aniso}.npy and <out>/site_<tag lower>.txt
(defaults: out = data/, tag = BCSmoke2y). Don't commit or share them -
PurpleAir's terms bar redistributing its data.
"""
import argparse
import io
import json
import math
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from build_bcsmoke_pilot import (  # noqa: E402  (shared constants/helpers)
    CELL_DLAT, CELL_DLON, D_ALONG, D_CROSS, FIRE_BBOX, FIRE_CELL_DEG, ISO_SCALE_KM,
    MAX_CANDIDATES, MAX_LAG_H, NODE_BBOX, SEED, WEATHER_VARS,
    bearing_rad, haversine_km, http_json, pa_key)

PA_CHUNK_DAYS = 60
PA_SLEEP = 1.0
GRID_DEG = 2.0                     # fire-source wind grid
STATIC_MIN_DAYS = 10               # Nov-Mar detection days that mark a static source
OM_PER_MIN, OM_PER_HOUR = 500, 4500  # stay under Open-Meteo's 600/min, 5000/h
FIRMS = "https://firms.modaps.eosdis.nasa.gov"

# set by main() from the CLI
T0 = T1 = HOURS = CACHE = None


def cache_dir(*parts):
    d = os.path.join(CACHE, *parts)
    os.makedirs(d, exist_ok=True)
    return d


def http_text(url, retries=4):
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=300) as r:
                return r.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503) and attempt < retries - 1:
                time.sleep(15 * (attempt + 1))
                continue
            raise


# ---------------------------------------------------------------- PurpleAir
def pa_sensors(key):
    fp = os.path.join(CACHE, "sensors.csv")
    if not os.path.exists(fp):
        b = NODE_BBOX
        p = dict(fields="latitude,longitude,date_created,last_seen", location_type=0, max_age=0,
                 nwlng=b["lon0"], nwlat=b["lat1"], selng=b["lon1"], selat=b["lat0"])
        j = http_json("https://api.purpleair.com/v1/sensors?" + urllib.parse.urlencode(p),
                      headers={"X-API-Key": key})
        pd.DataFrame(j["data"], columns=j["fields"]).to_csv(fp, index=False)
    s = pd.read_csv(fp)
    for c in ("date_created", "last_seen"):
        s[c] = pd.to_datetime(s[c], unit="s", utc=True)
    return s


def pa_history(key, sid, average):
    """pm2.5_alt for one sensor over [T0, T1] at `average` minutes, chunked."""
    parts = []
    for c0 in pd.date_range(T0, T1, freq=f"{PA_CHUNK_DAYS}D"):
        c1 = min(c0 + pd.Timedelta(days=PA_CHUNK_DAYS), T1 + pd.Timedelta(hours=1))
        fp = os.path.join(cache_dir("pa"), f"{sid}_{average}_{c0:%Y%m%d}.csv")
        if not os.path.exists(fp):
            p = dict(fields="pm2.5_alt", average=average,
                     start_timestamp=int(c0.timestamp()), end_timestamp=int(c1.timestamp()))
            try:
                j = http_json(f"https://api.purpleair.com/v1/sensors/{sid}/history?"
                              + urllib.parse.urlencode(p), headers={"X-API-Key": key})
                ti, pi = j["fields"].index("time_stamp"), j["fields"].index("pm2.5_alt")
                rows = [(r[ti], r[pi]) for r in j["data"]]
            except urllib.error.HTTPError as e:
                if e.code not in (400, 404):
                    raise
                rows = []
            h = pd.DataFrame(rows, columns=["time_stamp", "pm"])
            h["time_stamp"] = pd.to_datetime(h["time_stamp"].astype(float), unit="s", utc=True)
            h.to_csv(fp, index=False)
            time.sleep(PA_SLEEP)
        h = pd.read_csv(fp)
        if len(h):
            parts.append(pd.Series(h["pm"].values, index=pd.to_datetime(h["time_stamp"], utc=True)))
    if not parts:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([], tz="UTC"))
    s = pd.concat(parts)
    s = s[~s.index.duplicated()].sort_index()
    return s.where((s >= 0) & (s <= 1000)).dropna()


def daily_completeness(s):
    days = pd.date_range(T0.normalize(), T1.normalize(), freq="D")
    return s.index.floor("D").unique().isin(days).sum() / len(days)


def hourly_completeness(s):
    return s.reindex(HOURS).notna().mean()


def select_nodes(key, min_comp):
    fp = os.path.join(CACHE, "nodes.csv")
    if os.path.exists(fp):
        return pd.read_csv(fp)
    sensors = pa_sensors(key)
    b = NODE_BBOX
    cand = sensors[(sensors.date_created <= T0) & (sensors.last_seen >= T1)
                   & sensors.latitude.between(b["lat0"], b["lat1"])
                   & sensors.longitude.between(b["lon0"], b["lon1"])].copy()
    cand["cell"] = list(zip(((cand.latitude - b["lat0"]) / CELL_DLAT).astype(int),
                            ((cand.longitude - b["lon0"]) / CELL_DLON).astype(int)))
    cand = cand.sample(frac=1.0, random_state=SEED)
    cells = sorted(cand.cell.unique())
    print(f"{len(cand)} sensors alive across the window, in {len(cells)} cells")

    rows, n_hourly = [], 0
    for n, cell in enumerate(cells, 1):
        probe = [(daily_completeness(pa_history(key, r.sensor_index, 1440)), r)
                 for r in cand[cand.cell == cell].head(MAX_CANDIDATES).itertuples()]
        for dcomp, r in sorted(probe, key=lambda x: -x[0]):
            if dcomp < min_comp:
                break
            n_hourly += 1
            comp = hourly_completeness(pa_history(key, r.sensor_index, 60))
            if comp >= min_comp:
                rows.append(dict(site_id=r.sensor_index, lat=r.latitude, lon=r.longitude,
                                 completeness=round(comp, 3)))
                break
        print(f"\rcells {n}/{len(cells)}  nodes {len(rows)}  hourly pulls {n_hourly}", end="", flush=True)
    print()
    nodes = pd.DataFrame(rows)
    nodes.to_csv(fp, index=False)
    return nodes


def pm_matrix(key, nodes):
    cols, longest = [], []
    for sid in nodes.site_id:
        s = pa_history(key, sid, 60).reindex(HOURS)
        gap = s.isna().astype(int)
        longest.append(int(gap.groupby((gap == 0).cumsum()).sum().max()))
        cols.append(s.interpolate(method="time").ffill().bfill().to_numpy())
    pm = np.stack(cols, axis=1)
    interp = 1 - nodes.completeness.mean()
    print(f"PM2.5: {pm.shape}, interpolated ~{interp:.1%} of (hour, node) values; "
          f"longest gap per node: median {np.median(longest):.0f} h, max {max(longest)} h")
    return pm


# ---------------------------------------------------------------- weather
class OpenMeteo:
    """One-location requests, throttled to Open-Meteo's weighted free limits."""

    def __init__(self):
        self.log = []

    def _wait(self, w):
        while True:
            now = time.time()
            self.log = [(t, x) for t, x in self.log if now - t < 3600]
            minute = sum(x for t, x in self.log if now - t < 60)
            hour = sum(x for _, x in self.log)
            if minute + w <= OM_PER_MIN and hour + w <= OM_PER_HOUR:
                self.log.append((now, w))
                return
            time.sleep(5)

    def point(self, lat, lon, hourly, start, end, sub):
        fp = os.path.join(cache_dir(sub), f"{lat:.4f}_{lon:.4f}.json")
        if os.path.exists(fp):
            return json.load(open(fp))
        days = (pd.Timestamp(end) - pd.Timestamp(start)).days + 1
        w = max(1.0, days / 14) * max(1.0, len(hourly) / 10)
        p = dict(latitude=f"{lat:.4f}", longitude=f"{lon:.4f}", start_date=start, end_date=end,
                 hourly=",".join(hourly), wind_speed_unit="ms", models="era5", timezone="GMT")
        url = "https://archive-api.open-meteo.com/v1/archive?" + urllib.parse.urlencode(p)
        for attempt in range(30):
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
        out = dict(elevation=j.get("elevation", 0.0), hourly=j["hourly"])
        json.dump(out, open(fp, "w"))
        return out

    def many(self, lats, lons, hourly, start, end, sub):
        out = []
        for i, (a, b) in enumerate(zip(lats, lons), 1):
            out.append(self.point(a, b, hourly, start, end, sub))
            print(f"\rOpen-Meteo {sub}: {i}/{len(lats)}", end="", flush=True)
        print()
        return out


def meteo_to_array(meteo, hours, var):
    idx = pd.to_datetime(meteo[0]["hourly"]["time"]).tz_localize("UTC")
    a = np.array([m["hourly"][var] for m in meteo], dtype=float).T  # [time, loc]
    return pd.DataFrame(a, index=idx).reindex(hours).interpolate().ffill().bfill().to_numpy()


# ---------------------------------------------------------------- fires
def firms_api(key, d0, d1):
    avail = pd.read_csv(io.StringIO(http_text(f"{FIRMS}/api/data_availability/csv/{key}/ALL")))
    avail = avail.set_index("data_id")
    sp_max = pd.Timestamp(avail.loc["VIIRS_SNPP_SP", "max_date"])
    nrt_min = pd.Timestamp(avail.loc["VIIRS_SNPP_NRT", "min_date"])
    segments = []
    if d0 <= sp_max:
        segments.append(("VIIRS_SNPP_SP", d0, min(d1, sp_max)))
    if d1 > sp_max:
        s = max(d0, sp_max + pd.Timedelta(days=1))
        if s < nrt_min:
            print(f"WARNING: no FIRMS S-NPP data between {s.date()} and {nrt_min.date()}")
        segments.append(("VIIRS_SNPP_NRT", max(s, nrt_min), d1))
    b = FIRE_BBOX
    area = f"{b['lon0']},{b['lat0']},{b['lon1']},{b['lat1']}"
    frames = []
    for src, a, z in segments:
        print(f"FIRMS {src}: {a.date()} .. {z.date()}")
        for c in pd.date_range(a, z, freq="10D"):
            n = min(10, (z - c).days + 1)
            fp = os.path.join(cache_dir("firms"), f"{src}_{c:%Y%m%d}_{n}.csv")
            if not os.path.exists(fp):
                text = http_text(f"{FIRMS}/api/area/csv/{key}/{src}/{area}/{n}/{c:%Y-%m-%d}")
                if not text.startswith("latitude"):
                    raise RuntimeError(f"FIRMS API: {text[:300]}")
                open(fp, "w", encoding="utf-8").write(text)
                time.sleep(1.0)
            df = pd.read_csv(fp)
            dates = pd.to_datetime(df.acq_date)
            outside = ((dates < c) | (dates > c + pd.Timedelta(days=n - 1))).sum()
            if outside:
                print(f"WARNING: {fp} has {outside} rows outside its requested days")
            frames.append(df)
    return pd.concat(frames, ignore_index=True)


def firms_archives(d0, d1):
    frames = []
    for y in range(d0.year, d1.year + 1):
        for country in ("Canada", "United_States"):
            raw = os.path.join(cache_dir("firms"), f"viirs-snpp_{y}_{country}.csv")
            if not os.path.exists(raw):
                url = f"{FIRMS}/data/country/viirs-snpp/{y}/viirs-snpp_{y}_{country}.csv"
                print(f"downloading {url}")
                try:
                    urllib.request.urlretrieve(url, raw)
                except urllib.error.HTTPError:
                    sys.exit(f"FIRMS has no published {y} archive - set FIRMS_MAP_KEY "
                             "(free: https://firms.modaps.eosdis.nasa.gov/api/map_key/) to use the API.")
            frames.append(pd.read_csv(raw))
    return pd.concat(frames, ignore_index=True)


def load_fires(firms_key):
    fp = os.path.join(CACHE, "fires_cells_daily.csv")
    if os.path.exists(fp):
        return pd.read_csv(fp, parse_dates=["date"])
    d0 = (T0 - pd.Timedelta(hours=MAX_LAG_H)).tz_localize(None).normalize()
    d1 = T1.tz_localize(None).normalize()
    f = firms_api(firms_key, d0, d1) if firms_key else firms_archives(d0, d1)
    b = FIRE_BBOX
    f = f[f.latitude.between(b["lat0"], b["lat1"]) & f.longitude.between(b["lon0"], b["lon1"])
          & (f.confidence.astype(str).str[0].str.lower() != "l")]
    if "type" in f.columns:
        f = f[f["type"] == 0]                        # presumed vegetation fire
    f = f.drop_duplicates(["latitude", "longitude", "acq_date", "acq_time"])
    f["date"] = pd.to_datetime(f.acq_date)
    f = f[(f.date >= d0) & (f.date <= d1)].copy()
    f["clat"] = (np.floor(f.latitude / FIRE_CELL_DEG) + 0.5) * FIRE_CELL_DEG
    f["clon"] = (np.floor(f.longitude / FIRE_CELL_DEG) + 0.5) * FIRE_CELL_DEG

    winter = f[f.date.dt.month.isin([11, 12, 1, 2, 3])]
    off_season = winter.groupby(["clat", "clon"]).date.nunique()
    static = set(off_season[off_season >= STATIC_MIN_DAYS].index)
    keep = [(a, c) not in static for a, c in zip(f.clat, f.clon)]
    print(f"FIRMS: {len(f)} detections; dropping {len(static)} static-source cells "
          f"({(~np.array(keep)).sum()} detections)")
    f = f[keep]
    daily = f.groupby(["clat", "clon", "date"], as_index=False)["frp"].sum()
    daily.to_csv(fp, index=False)
    return daily


def fire_features(nodes, daily, om):
    ext_hours = pd.date_range(T0 - pd.Timedelta(hours=MAX_LAG_H), T1, freq="h")
    E, N, L = len(ext_hours), len(nodes), MAX_LAG_H
    cells = daily[["clat", "clon"]].drop_duplicates().reset_index(drop=True)
    cell_id = {(a, b): k for k, (a, b) in enumerate(zip(cells.clat, cells.clon))}
    print(f"fire cells: {len(cells)}, fire cell-days: {len(daily)}, emission hours: {E}")

    # wind at the fire source: nearest point of a GRID_DEG grid over FIRE_BBOX
    b = FIRE_BBOX
    glats = np.arange(b["lat0"] + GRID_DEG / 2, b["lat1"], GRID_DEG)
    glons = np.arange(b["lon0"] + GRID_DEG / 2, b["lon1"], GRID_DEG)
    gl = [(a, c) for a in glats for c in glons]
    meteo = om.many([p[0] for p in gl], [p[1] for p in gl], ["wind_speed_10m", "wind_direction_10m"],
                    str(ext_hours[0].date()), str(ext_hours[-1].date()), "meteo_grid")
    speed_kmh = 3.6 * meteo_to_array(meteo, ext_hours, "wind_speed_10m")       # [E, G]
    travel = np.radians(meteo_to_array(meteo, ext_hours, "wind_direction_10m") + 180.0)
    gi = np.clip(np.rint((cells.clat.values - glats[0]) / GRID_DEG), 0, len(glats) - 1).astype(int)
    gj = np.clip(np.rint((cells.clon.values - glons[0]) / GRID_DEG), 0, len(glons) - 1).astype(int)
    grid_of = gi * len(glons) + gj

    d = haversine_km(cells.clat.values[:, None], cells.clon.values[:, None],
                     nodes.lat.values[None, :], nodes.lon.values[None, :])        # [C, N]
    brg = bearing_rad(cells.clat.values[:, None], cells.clon.values[:, None],
                      nodes.lat.values[None, :], nodes.lon.values[None, :])
    K = np.exp(-d / ISO_SCALE_KM)

    day_pos = pd.Series(np.arange(E), index=ext_hours).groupby(
        ext_hours.normalize().tz_localize(None)).apply(list)
    t = np.arange(1, L + 1, dtype=float)[None, None, :]
    aniso = np.zeros((E + L, N))
    iso = np.zeros((E + L, N))
    for k, (date, grp) in enumerate(daily.groupby("date"), 1):
        if date not in day_pos.index:
            continue
        act = np.array([cell_id[(a, c)] for a, c in zip(grp.clat, grp.clon)])
        qa = grp.frp.values                                   # daily FRP, constant over the UTC day
        iso_k = qa @ K[act] / L                               # running 48 h mean, distance-decayed
        dA, bA, gA = d[act][..., None], brg[act], grid_of[act]
        for e in day_pos[date]:
            theta = (travel[e, gA][:, None] - bA)[..., None]
            x, y = dA * np.cos(theta), dA * np.sin(theta)
            v = speed_kmh[e, gA][:, None, None]
            G = (np.exp(-(x - v * t) ** 2 / (4 * D_ALONG * t) - y ** 2 / (4 * D_CROSS * t))
                 / (4 * math.pi * math.sqrt(D_ALONG * D_CROSS) * t))     # [A, N, L]
            aniso[e + 1:e + 1 + L] += np.einsum("a,anl->ln", qa, G)
            iso[e + 1:e + 1 + L] += iso_k
        if k % 25 == 0:
            print(f"\rfire days {k}", end="", flush=True)
    print()
    aniso, iso = aniso[L:L + len(HOURS)], iso[L:L + len(HOURS)]
    return np.log1p(1e3 * aniso), np.log1p(1e3 * iso)


# ---------------------------------------------------------------- main
def main():
    global T0, T1, HOURS, CACHE
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2024-10-01")
    ap.add_argument("--end", default="2026-09-30", help="last day, inclusive")
    ap.add_argument("--tag", default="BCSmoke2y")
    ap.add_argument("--cache", default=os.path.join(HERE, "cache_long"))
    ap.add_argument("--out", default=os.path.join(REPO, "data"))
    ap.add_argument("--min_completeness", type=float, default=0.85)
    ap.add_argument("--node_bbox", help="lat0,lat1,lon0,lon1 - override (e.g. a tiny test box)")
    args = ap.parse_args()

    T0 = pd.Timestamp(args.start, tz="UTC")
    T1 = pd.Timestamp(args.end, tz="UTC") + pd.Timedelta(hours=23)
    HOURS = pd.date_range(T0, T1, freq="h")
    CACHE = args.cache
    os.makedirs(CACHE, exist_ok=True)
    os.makedirs(args.out, exist_ok=True)
    if args.node_bbox:
        NODE_BBOX.update(zip(("lat0", "lat1", "lon0", "lon1"), map(float, args.node_bbox.split(","))))
    print(f"window {T0} .. {T1} ({len(HOURS)} h), cache {CACHE}")

    key = pa_key()
    nodes = select_nodes(key, args.min_completeness)
    print(f"{len(nodes)} nodes, median completeness {nodes.completeness.median():.1%}")
    pm = pm_matrix(key, nodes)

    om = OpenMeteo()
    meteo = om.many(nodes.lat.values, nodes.lon.values, WEATHER_VARS,
                    str(T0.date()), str(T1.date()), "meteo_nodes")
    weather = np.stack([meteo_to_array(meteo, HOURS, v) for v in WEATHER_VARS], axis=-1)  # [T,N,5]
    elev = [m["elevation"] for m in meteo]

    daily = load_fires(os.environ.get("FIRMS_MAP_KEY"))
    fire_aniso, fire_iso = fire_features(nodes, daily, om)

    variants = {"nofire": weather,
                "iso": np.concatenate([weather, fire_iso[..., None]], axis=-1),
                "aniso": np.concatenate([weather, fire_aniso[..., None]], axis=-1)}
    for name, feat in variants.items():
        arr = np.concatenate([feat, pm[..., None]], axis=-1).astype(np.float64)
        assert np.isfinite(arr).all(), name
        np.save(os.path.join(args.out, f"{args.tag}_{name}.npy"), arr)
        print(f"wrote {args.tag}_{name}.npy {arr.shape}")
    site_fp = os.path.join(args.out, f"site_{args.tag.lower()}.txt")
    with open(site_fp, "w") as f:
        for i, r in enumerate(nodes.itertuples()):
            f.write(f"{i} {r.site_id} {r.lon} {r.lat} {elev[i]}\n")
    print(f"wrote {site_fp}")

    pmf = pd.DataFrame(pm, index=HOURS)
    month = pmf.index.strftime("%Y-%m")
    monthly = pd.DataFrame({
        "pm_mean": pmf.mean(axis=1).groupby(month).mean(),
        "pm_p99": pmf.groupby(month).apply(lambda g: np.percentile(g.values, 99)),
        "node_hours_gt35.5": (pmf > 35.5).mean(axis=1).groupby(month).mean(),
        "hours_any_gt35.5": (pmf > 35.5).any(axis=1).groupby(month).mean(),
        "fire_aniso_mean": pd.Series(fire_aniso.mean(axis=1), index=HOURS).groupby(month).mean(),
    }).round(3)
    monthly.to_csv(os.path.join(args.out, f"{args.tag}_monthly.csv"))
    print(monthly.to_string())
    pm_flat = np.log1p(pm).ravel()
    for name, fv in (("fire_iso", fire_iso), ("fire_aniso", fire_aniso)):
        print(f"corr(log PM2.5, {name}) = {np.corrcoef(pm_flat, fv.ravel())[0, 1]:.3f}")


if __name__ == "__main__":
    main()
