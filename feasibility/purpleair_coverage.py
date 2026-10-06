"""Feasibility check (step 2): how many outdoor PurpleAir sensors in BC +
the US Pacific Northwest were alive across the 2021 and 2023 smoke seasons?

One /v1/sensors call over a bounding box, max_age=0 so sensors that have
since gone offline are included too (default max_age drops anything not
seen in the last week, which would hide exactly the historical sensors we
care about). Cost is roughly (#sensors x #fields) points - a few tens of
thousands at most, well inside the free 1M-point allowance.

"Covers a season" here means date_created <= season start AND last_seen >=
season end - a lifetime-overlap proxy, NOT a completeness check (a sensor
can be alive on both ends but offline for weeks in between). Actual
per-sensor completeness needs /history calls, which cost far more points;
do that only on a sample once this coarse count looks promising.

Usage:
    set PURPLEAIR_API_KEY=<your READ key>   (PowerShell: $env:PURPLEAIR_API_KEY="...")
    python feasibility/purpleair_coverage.py
Writes feasibility/purpleair_sensors.csv (raw list, for your own use only -
PurpleAir's terms bar redistributing it) and prints a summary.
"""
import json
import math
import os
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone

import numpy as np
import pandas as pd

API_URL = "https://api.purpleair.com/v1/sensors"

# BC + WA/OR/ID + western Alberta margin (lon, lat)
BBOX = dict(nwlng=-139.1, nwlat=60.0, selng=-110.0, selat=42.0)

SEASONS = {
    "2021 (Jul-Aug)": ("2021-07-01", "2021-08-31"),
    "2023 (May-Sep)": ("2023-05-01", "2023-09-30"),
}

FIELDS = ["name", "latitude", "longitude", "date_created", "last_seen", "location_type", "model"]

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def fetch_sensors(api_key):
    params = dict(fields=",".join(FIELDS), location_type=0, max_age=0, **BBOX)
    url = f"{API_URL}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"X-API-Key": api_key})
    with urllib.request.urlopen(req, timeout=120) as resp:
        payload = json.load(resp)
    df = pd.DataFrame(payload["data"], columns=payload["fields"])
    for col in ("date_created", "last_seen"):
        df[col] = pd.to_datetime(df[col], unit="s", utc=True)
    return df


def region_of(lat, lon):
    # Coarse: north of the 49th parallel and west of ~-114 is treated as BC
    # (the real BC/AB border follows the continental divide - fine for a count).
    if lat >= 49.0 and lon <= -114.0:
        return "BC"
    # southern Vancouver Island (Victoria) dips below 49N; the Juan de Fuca
    # strait (~48.3N) separates it from WA's Olympic Peninsula.
    if lat >= 48.3 and lon <= -123.2 and not (lat < 48.45 and lon < -124.3):
        return "BC"
    if lat >= 49.0:
        return "AB (margin)"
    if lat >= 45.55:
        return "WA / N-ID"
    return "OR / S-ID"


def median_nn_km(lat, lon):
    """Median nearest-neighbour distance (haversine), a density proxy."""
    if len(lat) < 2:
        return float("nan")
    la, lo = np.radians(lat)[:, None], np.radians(lon)[:, None]
    dlat, dlon = la - la.T, lo - lo.T
    a = np.sin(dlat / 2) ** 2 + np.cos(la) * np.cos(la.T) * np.sin(dlon / 2) ** 2
    d = 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    np.fill_diagonal(d, np.inf)
    return float(np.median(d.min(axis=1)))


def summarize(df):
    df = df.copy()
    df["region"] = [region_of(a, b) for a, b in zip(df["latitude"], df["longitude"])]
    print(f"Outdoor sensors ever registered in bbox: {len(df)}")
    print(df["region"].value_counts().to_string(), "\n")

    rows = []
    for season, (start, end) in SEASONS.items():
        s = pd.Timestamp(start, tz=timezone.utc)
        e = pd.Timestamp(end, tz=timezone.utc)
        alive = df[(df["date_created"] <= s) & (df["last_seen"] >= e)]
        for region, g in alive.groupby("region"):
            rows.append(dict(season=season, region=region, sensors=len(g),
                             median_nn_km=round(median_nn_km(g["latitude"].values,
                                                             g["longitude"].values), 1)))
        rows.append(dict(season=season, region="ALL", sensors=len(alive),
                         median_nn_km=round(median_nn_km(alive["latitude"].values,
                                                         alive["longitude"].values), 1)))
    table = pd.DataFrame(rows)
    print("Sensors alive across the whole season (lifetime-overlap proxy):")
    print(table.to_string(index=False))
    return table


def main():
    api_key = os.environ.get("PURPLEAIR_API_KEY")
    key_file = os.path.join(OUT_DIR, ".purpleair_key")
    if not api_key and os.path.exists(key_file):
        with open(key_file, encoding="utf-8") as f:
            api_key = f.read().strip()
    if not api_key:
        sys.exit("Set PURPLEAIR_API_KEY, or put your PurpleAir READ key in "
                 "feasibility/.purpleair_key (one line).")
    df = fetch_sensors(api_key)
    df.to_csv(os.path.join(OUT_DIR, "purpleair_sensors.csv"), index=False)
    table = summarize(df)
    table.to_csv(os.path.join(OUT_DIR, "purpleair_coverage_summary.csv"), index=False)
    print(f"\nRun at {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC; "
          f"raw list saved to feasibility/purpleair_sensors.csv")


if __name__ == "__main__":
    main()
