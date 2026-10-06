"""Build the BC wildfire-smoke PILOT datasets (dataset_num 5/6/7).

Question the pilot answers: does telling a forecaster where fires are help
it predict smoke PM2.5 at low-cost sensors - and does carrying that fire
signal with the WIND (AirLapseV2's own 2-D anisotropic Green's function)
help beyond simple "fires nearby" proximity?

Three otherwise-identical datasets, same nodes/hours/weather/PM2.5:
  BCSmoke_nofire.npy  [temperature, rh, rain, wind_speed10, wind_direction10, pm]
  BCSmoke_iso.npy     [... 5 weather ..., fire_iso,   pm]  distance-only fire signal
  BCSmoke_aniso.npy   [... 5 weather ..., fire_aniso, pm]  wind-transported fire signal

Sources (all free; only PurpleAir needs a key):
  - PM2.5: PurpleAir hourly pm2.5_alt (key from feasibility/.purpleair_key).
    One sensor per ~25 km cell in 48-52N, 124-117W (southern BC + northern
    WA), the first of up to 3 random candidates per cell with >= 85%
    hourly completeness over the window. Gaps linearly interpolated.
  - Weather: ERA5 hourly via the Open-Meteo archive API, at each node.
  - Fires: NASA FIRMS VIIRS S-NPP 2023 country archives (Canada + US),
    nominal/high confidence only, 45-58N / 130-110W, aggregated to 0.25 deg
    cells as daily summed FRP (MW), held constant over that UTC day.
    Wind at each active fire cell is also ERA5 via Open-Meteo.

Fire features at node i, hour t (48 h look-back, tau = 1..48 h):
  fire_aniso = log1p(1e3 * sum_c sum_tau q_c(t-tau) * G_ci(tau))
      G = 2-D anisotropic advection-diffusion Green's function, identical
      in form to AdaptivePhysicsTransport2D (model/airlapse_v2.py):
      downwind/crosswind decomposition along the fire cell's wind bearing
      at emission time, D_along=50, D_cross=20 km^2/h (that module's init
      defaults). Straight-line, wind frozen at the source per emission hour -
      a pilot-grade approximation, not a trajectory model.
  fire_iso = log1p(1e3 * sum_c mean_tau q_c(t-tau) * exp(-d_ci / 150 km))
      Same fires, same window, no wind - the "is proximity alone enough?"
      control.

Everything downloaded is cached in pilot/cache/ (git-ignored; PurpleAir's
terms bar redistributing its data), so re-runs cost no API points.
"""
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
CACHE = os.path.join(HERE, "cache")
DATA = os.path.join(REPO, "data")
os.makedirs(os.path.join(CACHE, "pa"), exist_ok=True)

T0 = pd.Timestamp("2023-07-01 00:00", tz="UTC")
T1 = pd.Timestamp("2023-08-31 23:00", tz="UTC")
HOURS = pd.date_range(T0, T1, freq="h")
MAX_LAG_H = 48

NODE_BBOX = dict(lat0=48.0, lat1=52.0, lon0=-124.0, lon1=-117.0)
CELL_DLAT, CELL_DLON = 0.225, 0.35          # ~25 km
MIN_COMPLETENESS = 0.85
MAX_CANDIDATES = 3
SEED = 0

FIRE_BBOX = dict(lat0=45.0, lat1=58.0, lon0=-130.0, lon1=-110.0)
FIRE_CELL_DEG = 0.25
D_ALONG, D_CROSS = 50.0, 20.0               # km^2/h
ISO_SCALE_KM = 150.0

WEATHER_VARS = ["temperature_2m", "relative_humidity_2m", "precipitation",
                "wind_speed_10m", "wind_direction_10m"]


# ---------------------------------------------------------------- helpers
def http_json(url, headers=None, retries=4):
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=headers or {})
            with urllib.request.urlopen(req, timeout=180) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503) and attempt < retries - 1:
                time.sleep(10 * (attempt + 1))
                continue
            raise


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def bearing_rad(lat1, lon1, lat2, lon2):
    """Compass bearing (0=N, clockwise) from point 1 to point 2, radians."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    y = np.sin(lon2 - lon1) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lon2 - lon1)
    return np.arctan2(y, x)


# ---------------------------------------------------------------- PurpleAir
def pa_key():
    key = os.environ.get("PURPLEAIR_API_KEY")
    kf = os.path.join(REPO, "feasibility", ".purpleair_key")
    if not key and os.path.exists(kf):
        key = open(kf, encoding="utf-8").read().strip()
    if not key:
        sys.exit("No PurpleAir key (env PURPLEAIR_API_KEY or feasibility/.purpleair_key).")
    return key


def pa_history(key, sid):
    fp = os.path.join(CACHE, "pa", f"{sid}.csv")
    if os.path.exists(fp):
        return pd.read_csv(fp, index_col=0, parse_dates=True)["pm"]
    p = dict(fields="pm2.5_alt", average=60,
             start_timestamp=int(T0.timestamp()),
             end_timestamp=int((T1 + pd.Timedelta(hours=1)).timestamp()))
    j = http_json(f"https://api.purpleair.com/v1/sensors/{sid}/history?{urllib.parse.urlencode(p)}",
                  headers={"X-API-Key": key})
    h = pd.DataFrame(j["data"], columns=j["fields"])
    s = pd.Series(h["pm2.5_alt"].values,
                  index=pd.to_datetime(h["time_stamp"], unit="s", utc=True), name="pm")
    s = s[~s.index.duplicated()].sort_index()
    s.to_frame().to_csv(fp)
    time.sleep(1.0)
    return s


def select_nodes():
    fp = os.path.join(CACHE, "nodes.csv")
    if os.path.exists(fp):
        return pd.read_csv(fp)
    key = pa_key()
    sensors = pd.read_csv(os.path.join(REPO, "feasibility", "purpleair_sensors.csv"),
                          parse_dates=["date_created", "last_seen"])
    b = NODE_BBOX
    cand = sensors[(sensors.date_created <= T0) & (sensors.last_seen >= T1)
                   & sensors.latitude.between(b["lat0"], b["lat1"])
                   & sensors.longitude.between(b["lon0"], b["lon1"])].copy()
    cand["cell"] = list(zip(((cand.latitude - b["lat0"]) / CELL_DLAT).astype(int),
                            ((cand.longitude - b["lon0"]) / CELL_DLON).astype(int)))
    cand = cand.sample(frac=1.0, random_state=SEED)

    rows, calls = [], 0
    cells = sorted(cand.cell.unique())
    for n, cell in enumerate(cells, 1):
        for r in cand[cand.cell == cell].head(MAX_CANDIDATES).itertuples():
            s = pa_history(key, r.sensor_index)
            calls += 1
            s = s[(s >= 0) & (s <= 1000)].reindex(HOURS)
            comp = s.notna().mean()
            if comp >= MIN_COMPLETENESS:
                rows.append(dict(site_id=r.sensor_index, lat=r.latitude, lon=r.longitude,
                                 completeness=round(comp, 3)))
                break
        print(f"\rcells {n}/{len(cells)}  nodes {len(rows)}  PurpleAir calls {calls}", end="")
    print()
    nodes = pd.DataFrame(rows)
    nodes.to_csv(fp, index=False)
    return nodes


def pm_matrix(nodes):
    cols, n_interp = [], 0
    for sid in nodes.site_id:
        s = pa_history(None, sid)  # cached by select_nodes()
        s = s.where((s >= 0) & (s <= 1000)).reindex(HOURS)
        n_interp += s.isna().sum()
        cols.append(s.interpolate(method="time").ffill().bfill().to_numpy())
    pm = np.stack(cols, axis=1)
    print(f"PM2.5: {pm.shape}, interpolated {n_interp / pm.size:.1%} of (hour, node) values")
    return pm


# ---------------------------------------------------------------- weather
def open_meteo(lats, lons, start, end, tag, hourly=WEATHER_VARS, batch=50):
    fp = os.path.join(CACHE, f"meteo_{tag}.json")
    if os.path.exists(fp):
        return json.load(open(fp))
    out = []
    for i in range(0, len(lats), batch):
        p = dict(latitude=",".join(f"{x:.4f}" for x in lats[i:i + batch]),
                 longitude=",".join(f"{x:.4f}" for x in lons[i:i + batch]),
                 start_date=start, end_date=end, hourly=",".join(hourly),
                 wind_speed_unit="ms", models="era5", timezone="GMT")
        j = http_json("https://archive-api.open-meteo.com/v1/archive?" + urllib.parse.urlencode(p))
        out.extend(j if isinstance(j, list) else [j])
        print(f"\rOpen-Meteo {tag}: {len(out)}/{len(lats)}", end="")
        time.sleep(2.0)
    print()
    json.dump(out, open(fp, "w"))
    return out


def meteo_to_array(meteo, hours, var):
    idx = pd.to_datetime(meteo[0]["hourly"]["time"]).tz_localize("UTC")
    a = np.array([m["hourly"][var] for m in meteo], dtype=float).T  # [time, loc]
    df = pd.DataFrame(a, index=idx).reindex(hours)
    return df.interpolate().ffill().bfill().to_numpy()


# ---------------------------------------------------------------- fires
def load_fires():
    fp = os.path.join(CACHE, "fires_cells_daily.csv")
    if os.path.exists(fp):
        return pd.read_csv(fp, parse_dates=["date"])
    frames = []
    for country in ("Canada", "United_States"):
        raw = os.path.join(CACHE, f"viirs-snpp_2023_{country}.csv")
        if not os.path.exists(raw):
            url = f"https://firms.modaps.eosdis.nasa.gov/data/country/viirs-snpp/2023/viirs-snpp_2023_{country}.csv"
            print(f"downloading FIRMS {country} ...")
            urllib.request.urlretrieve(url, raw)
        df = pd.read_csv(raw, usecols=["latitude", "longitude", "acq_date", "confidence", "frp", "type"])
        b = FIRE_BBOX
        df = df[df.latitude.between(b["lat0"], b["lat1"]) & df.longitude.between(b["lon0"], b["lon1"])
                & (df.confidence != "l") & (df.type == 0)]   # type 0 = presumed vegetation fire
        frames.append(df)
    f = pd.concat(frames)
    f["date"] = pd.to_datetime(f.acq_date)
    f = f[(f.date >= (T0 - pd.Timedelta(hours=MAX_LAG_H)).tz_localize(None).normalize())
          & (f.date <= T1.tz_localize(None).normalize())]
    f["clat"] = (np.floor(f.latitude / FIRE_CELL_DEG) + 0.5) * FIRE_CELL_DEG
    f["clon"] = (np.floor(f.longitude / FIRE_CELL_DEG) + 0.5) * FIRE_CELL_DEG
    daily = f.groupby(["clat", "clon", "date"], as_index=False)["frp"].sum()
    daily.to_csv(fp, index=False)
    return daily


def fire_features(nodes, daily):
    ext_hours = pd.date_range(T0 - pd.Timedelta(hours=MAX_LAG_H), T1, freq="h")
    cells = daily[["clat", "clon"]].drop_duplicates().reset_index(drop=True)
    C, N, E = len(cells), len(nodes), len(ext_hours)
    print(f"fire cells: {C}, emission hours: {E}")

    # q[e, c]: daily FRP held constant over the UTC day
    cell_id = {(a, b): k for k, (a, b) in enumerate(zip(cells.clat, cells.clon))}
    day_idx = pd.Series(np.arange(E), index=ext_hours).groupby(ext_hours.normalize().tz_localize(None)).apply(list)
    q = np.zeros((E, C))
    for r in daily.itertuples():
        if r.date in day_idx.index:
            q[day_idx[r.date], cell_id[(r.clat, r.clon)]] += r.frp

    meteo = open_meteo(cells.clat.values, cells.clon.values,
                       str(ext_hours[0].date()), str(ext_hours[-1].date()), "fires",
                       hourly=["wind_speed_10m", "wind_direction_10m"], batch=100)
    speed_kmh = 3.6 * meteo_to_array(meteo, ext_hours, "wind_speed_10m")        # [E, C]
    travel = np.radians(meteo_to_array(meteo, ext_hours, "wind_direction_10m") + 180.0)

    d = haversine_km(cells.clat.values[:, None], cells.clon.values[:, None],
                     nodes.lat.values[None, :], nodes.lon.values[None, :])        # [C, N]
    brg = bearing_rad(cells.clat.values[:, None], cells.clon.values[:, None],
                      nodes.lat.values[None, :], nodes.lon.values[None, :])       # [C, N]

    taus = np.arange(1, MAX_LAG_H + 1, dtype=float)                                # [L]
    aniso = np.zeros((E + MAX_LAG_H, N))
    iso_q = np.zeros((E + MAX_LAG_H, C))
    for e in range(E):
        active = np.nonzero(q[e])[0]
        if active.size == 0:
            continue
        qa = q[e, active]
        theta = travel[e, active][:, None] - brg[active]                           # [A, N]
        x = (d[active] * np.cos(theta))[..., None]                                 # [A, N, 1]
        y = (d[active] * np.sin(theta))[..., None]
        v = speed_kmh[e, active][:, None, None]
        t = taus[None, None, :]
        G = (np.exp(-(x - v * t) ** 2 / (4 * D_ALONG * t) - y ** 2 / (4 * D_CROSS * t))
             / (4 * math.pi * math.sqrt(D_ALONG * D_CROSS) * t))                   # [A, N, L]
        contrib = np.einsum("a,anl->ln", qa, G)                                    # [L, N]
        aniso[e + 1:e + 1 + MAX_LAG_H] += contrib
        iso_q[e + 1:e + 1 + MAX_LAG_H, active] += qa / MAX_LAG_H                   # running 48h mean

    off = MAX_LAG_H  # ext_hours[off] == T0
    aniso = aniso[off:off + len(HOURS)]
    iso = iso_q[off:off + len(HOURS)] @ np.exp(-d / ISO_SCALE_KM)                  # [T, N]
    return np.log1p(1e3 * aniso), np.log1p(1e3 * iso)


# ---------------------------------------------------------------- main
def main():
    nodes = select_nodes()
    print(f"{len(nodes)} nodes, median completeness {nodes.completeness.median():.0%}")
    pm = pm_matrix(nodes)

    meteo = open_meteo(nodes.lat.values, nodes.lon.values,
                       str(T0.date()), str(T1.date()), "nodes")
    weather = np.stack([meteo_to_array(meteo, HOURS, v) for v in WEATHER_VARS], axis=-1)  # [T,N,5]
    elev = [m.get("elevation", 0.0) for m in meteo]

    daily = load_fires()
    fire_aniso, fire_iso = fire_features(nodes, daily)

    base = weather
    variants = {
        "nofire": base,
        "iso": np.concatenate([base, fire_iso[..., None]], axis=-1),
        "aniso": np.concatenate([base, fire_aniso[..., None]], axis=-1),
    }
    for name, feat in variants.items():
        arr = np.concatenate([feat, pm[..., None]], axis=-1).astype(np.float64)
        assert np.isfinite(arr).all(), name
        np.save(os.path.join(DATA, f"BCSmoke_{name}.npy"), arr)
        print(f"wrote data/BCSmoke_{name}.npy {arr.shape}")

    with open(os.path.join(DATA, "site_bcsmoke.txt"), "w") as f:
        for i, r in enumerate(nodes.itertuples()):
            f.write(f"{i} {r.site_id} {r.lon} {r.lat} {elev[i]}\n")
    print("wrote data/site_bcsmoke.txt")

    # quick sanity: does the fire signal co-move with PM2.5 at all?
    pm_flat = np.log1p(pm).ravel()
    for name, fv in (("fire_iso", fire_iso), ("fire_aniso", fire_aniso)):
        print(f"corr(log PM2.5, {name}) = {np.corrcoef(pm_flat, fv.ravel())[0, 1]:.3f}")
    print(f"PM2.5 mean {pm.mean():.1f}, p99 {np.percentile(pm, 99):.1f}, "
          f"hours>35.5 (any node) {(pm > 35.5).any(axis=1).mean():.0%}")


if __name__ == "__main__":
    main()
