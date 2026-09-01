"""Build USAir.npy + city_us.txt from data/US-modified.csv.

Converts the long-format US state-capitals CSV (one row per city per day:
date, city, lat, lon, temperature_c, humidity_percent, wind_speed_mps, ...,
pm25) into the same (time, node, feature) tensor layout dataset.py already
expects from KnowAir.npy - last feature column is pm25, everything before
it is the per-node feature vector - so no change to dataset.py's
_load_npy() is needed, only to what path it loads and how the remaining
per-family feature processing works (see dataset.py's `family == 'us'`
branch in _process_feature).

Output:
  data/USAir.npy    - float64 array, shape (T, N, 4):
                       [temperature_c, humidity_percent, wind_speed_mps, pm25]
  data/city_us.txt  - "idx city lon lat", one line per node, same order as
                       USAir.npy's node axis (alphabetical by city name -
                       pandas pivot's default column sort, kept explicit
                       here so the mapping can't silently drift if pandas'
                       default ever changed).

Known limitations, both accepted per the project owner's call (diversity
of geography over strict comparability with KnowAir) and documented again
at their point of use in dataset.py/graph.py:
  - Daily resolution (not KnowAir's 3-hourly).
  - No wind direction in the source data (only a scalar speed) - graph
    models that use wind direction for advection (PM25_GNN-style) get a
    constant placeholder direction instead of a real one.
  - No altitude/elevation source wired up for these 51 stations - graph
    edges are built from lon/lat only, and AirLapse's station_elevation
    input is zero-filled for this dataset.
"""
import os

import numpy as np
import pandas as pd

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FP = os.path.join(DATA_DIR, 'US-modified.csv')
NPY_FP = os.path.join(DATA_DIR, 'USAir.npy')
CITY_FP = os.path.join(DATA_DIR, 'city_us.txt')

# Order must match dataset.py's US_METERO_VAR.
FEATURE_COLS = ['temperature_c', 'humidity_percent', 'wind_speed_mps']


def main():
    df = pd.read_csv(CSV_FP)
    df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')

    cities = sorted(df['city'].unique())
    dates = sorted(df['date'].unique())
    print(f'{len(cities)} cities x {len(dates)} days = {len(cities) * len(dates)} points '
          f'(csv has {len(df)} rows)')
    assert len(cities) * len(dates) == len(df), \
        'expected one row per (city, date) - CSV has gaps or duplicates'

    # One static lon/lat per city - sanity-check it really is static before
    # collapsing it away (a city with multiple recorded coordinates would
    # silently pick the wrong one otherwise).
    latlon = df.groupby('city')[['latitude', 'longitude']].nunique()
    assert (latlon == 1).all().all(), 'expected one fixed lat/lon per city'
    city_coord = df.groupby('city')[['longitude', 'latitude']].first()

    cols = FEATURE_COLS + ['pm25']
    arr = np.zeros((len(dates), len(cities), len(cols)), dtype=np.float64)
    for j, city in enumerate(cities):
        sub = df[df['city'] == city].sort_values('date')
        assert list(sub['date']) == dates, f'{city}: missing/unordered dates'
        arr[:, j, :] = sub[cols].to_numpy()

    np.save(NPY_FP, arr)
    print(f'wrote {NPY_FP}: shape={arr.shape} (time, node, feature) dtype={arr.dtype}')
    print(f'  features: {FEATURE_COLS} + [pm25]')
    print(f'  time range: {dates[0].date()} .. {dates[-1].date()}, daily')

    with open(CITY_FP, 'w') as f:
        for idx, city in enumerate(cities):
            lon, lat = city_coord.loc[city]
            f.write(f'{idx} {city} {lon} {lat}\n')
    print(f'wrote {CITY_FP}: {len(cities)} nodes')


if __name__ == '__main__':
    main()
