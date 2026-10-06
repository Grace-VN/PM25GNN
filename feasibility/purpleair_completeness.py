"""Feasibility check (step 2b): actual hourly data completeness for a random
sample of BC PurpleAir sensors that were 'alive' through summer 2023.

Follows purpleair_coverage.py (which only checked install/last-seen dates).
Samples N sensors (seeded), pulls hourly pm2.5_alt for August 2023 (BC's
worst smoke month) in two calls per sensor (the API caps 60-min-average
history at ~2 weeks per call), and reports the fraction of the 744 hours
with a plausible reading.

Reads feasibility/purpleair_sensors.csv (written by purpleair_coverage.py)
and the key from feasibility/.purpleair_key or PURPLEAIR_API_KEY. Writes
feasibility/purpleair_completeness.csv (raw per-sensor stats - keep out of
git like the other CSVs; see .gitignore).
"""
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import purpleair_coverage as pc

N_SAMPLE = 50
SEED = 0
START = pd.Timestamp("2023-08-01", tz="UTC")
END = pd.Timestamp("2023-09-01", tz="UTC")
EXPECTED_HOURS = int((END - START) / pd.Timedelta(hours=1))  # 744
OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_key():
    key = os.environ.get("PURPLEAIR_API_KEY")
    kf = os.path.join(OUT_DIR, ".purpleair_key")
    if not key and os.path.exists(kf):
        key = open(kf, encoding="utf-8").read().strip()
    if not key:
        sys.exit("No PurpleAir key found (env var or feasibility/.purpleair_key).")
    return key


def history(key, sensor_index, t0, t1):
    params = dict(fields="pm2.5_alt", average=60,
                  start_timestamp=t0.strftime("%Y-%m-%dT%H:%M:%SZ"),
                  end_timestamp=t1.strftime("%Y-%m-%dT%H:%M:%SZ"))
    url = f"https://api.purpleair.com/v1/sensors/{sensor_index}/history?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"X-API-Key": key})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                j = json.load(resp)
            return pd.DataFrame(j["data"], columns=j["fields"])
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(5 * (attempt + 1))
                continue
            raise
    raise RuntimeError(f"rate-limited repeatedly on sensor {sensor_index}")


def main():
    key = load_key()
    df = pd.read_csv(os.path.join(OUT_DIR, "purpleair_sensors.csv"),
                     parse_dates=["date_created", "last_seen"])
    df["region"] = [pc.region_of(a, b) for a, b in zip(df["latitude"], df["longitude"])]
    alive = df[(df["region"] == "BC")
               & (df["date_created"] <= pd.Timestamp("2023-05-01", tz="UTC"))
               & (df["last_seen"] >= pd.Timestamp("2023-09-30", tz="UTC"))]
    sample = alive.sample(n=min(N_SAMPLE, len(alive)), random_state=SEED)

    mid = START + (END - START) / 2
    rows = []
    for i, r in enumerate(sample.itertuples(), 1):
        try:
            parts = [history(key, r.sensor_index, START, mid),
                     history(key, r.sensor_index, mid, END)]
            h = pd.concat(parts).drop_duplicates("time_stamp")
            v = h["pm2.5_alt"]
            good = v.notna() & (v >= 0) & (v <= 1000)
            ts = h.loc[good, "time_stamp"]
            # the API echoes the request's timestamp format: ISO strings when
            # start/end are ISO (as here), Unix seconds otherwise
            hours = (pd.to_datetime(ts, unit="s", utc=True)
                     if pd.api.types.is_numeric_dtype(ts) else pd.to_datetime(ts, utc=True))
            hours = hours[(hours >= START) & (hours < END)]
            comp = hours.nunique() / EXPECTED_HOURS
        except Exception as e:  # keep going; count as failed, not as 0% complete
            print(f"[{i}/{len(sample)}] sensor {r.sensor_index}: FAILED ({e})")
            rows.append(dict(sensor_index=r.sensor_index, lat=r.latitude, lon=r.longitude,
                             completeness=np.nan))
            continue
        rows.append(dict(sensor_index=r.sensor_index, lat=r.latitude, lon=r.longitude,
                         completeness=comp))
        print(f"[{i}/{len(sample)}] sensor {r.sensor_index}: {comp:.0%}")
        time.sleep(1.0)

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT_DIR, "purpleair_completeness.csv"), index=False)
    ok = res.dropna(subset=["completeness"])
    print(f"\nSampled {len(res)} BC sensors, {len(ok)} queried successfully "
          f"(Aug 2023, {EXPECTED_HOURS} hourly slots each)")
    print(ok["completeness"].describe().round(2).to_string())
    for thr in (0.5, 0.8, 0.9):
        print(f"  >= {thr:.0%} complete: {(ok['completeness'] >= thr).mean():.0%} of sampled sensors")
    print(f"  north of 52N: {(ok['lat'] > 52).sum()} sensors, "
          f"median completeness {ok.loc[ok['lat'] > 52, 'completeness'].median():.0%}")


if __name__ == "__main__":
    main()
