"""Build SensorAir.npy + site_sensor.txt from data/data.csv.

data.csv is a ~2.27M-row hourly low-cost PM2.5 sensor feed (6,031
site_ids, Sep 2022 - Jun 2023, North America), shaped very differently
from KnowAir: most "sites" are short-lived (median site reports only
~260 of the ~5,979 possible hours - consistent with a crowd-sourced
network where individual sensors join/drop rather than a fixed permanent
grid), so a dense (time, node, feature) tensor can't be built from the
whole thing without either a tiny node count or heavy fabrication.

What this script actually does instead: picks the single best window/
site-set combination found by inspection (see the investigation in this
conversation - not re-derived here) rather than a generic "handle any
input" pipeline:

  - Window: 2023-05-14 00:00 -> 2023-06-02 16:00 (473 hours, ~19.7 days).
    A large batch of new sensors joined the network on 2023-05-13 (daily
    site count jumps from ~718 to 1,422 that day - see the exploration
    history) and reports near-continuously afterward; before this window
    coverage collapses (e.g. 0 sites clear 50% density over any 45+ day
    span). This is the tail end of the collection period, not an
    arbitrary slice.
  - Sites: the 197 with >=95% hourly coverage in that window. Their
    coverage clusters tightly around 96% with a near-identical ~19-20
    hour missing run each (see investigation) - a single shared outage
    hitting the whole network at once, not per-site flakiness - so it's
    interpolated below rather than treated as many small independent
    gaps.

Output:
  data/SensorAir.npy  - float64 array, shape (473, 197, 6):
                         [temperature, relative_humidity, rain,
                          wind_speed10, wind_direction10, pm]
  data/site_sensor.txt - "idx site_id lon lat elevation", one line per
                         node, same order as SensorAir.npy's node axis.
                         The trailing elevation field (absent from
                         data/city.txt's format) is real per-site
                         elevation from the source data - graph.py reads
                         it directly instead of the raster lookup used
                         for the China city set (see Graph._gen_nodes).

Known limitations (accepted - real crowd-sourced sensor data, not a
curated benchmark - documented again at point of use in dataset.py):
  - ~19.7 days total - far shorter than KnowAir's shortest sub-dataset.
  - The shared ~20-hour outage is filled by linear time-interpolation
    for every column, including pm itself - about 4% of (site, hour)
    values in the output are interpolated, not measured.
  - wind_direction10 is interpolated the same (linear) way, which isn't
    physically correct across the 0/360 discontinuity (e.g. 350deg and
    10deg averages to 180deg, not 0deg) - a real inaccuracy right at the
    handful of interpolated hours, accepted rather than special-cased
    for one ~20-hour gap out of 473.
  - PM values are raw/uncalibrated low-cost-sensor readings (this
    dataset's max is 4,835 ug/m3 vs a mean of ~13); no correction factor
    is applied here.
"""
import os

import numpy as np
import pandas as pd

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FP = os.path.join(DATA_DIR, 'data.csv')
NPY_FP = os.path.join(DATA_DIR, 'SensorAir.npy')
SITE_FP = os.path.join(DATA_DIR, 'site_sensor.txt')

WINDOW_START = pd.Timestamp('2023-05-14 00:00')
WINDOW_END = pd.Timestamp('2023-06-02 16:00')
COVERAGE_THRESHOLD = 0.95

# Order must match dataset.py's SENSOR_METERO_VAR.
FEATURE_COLS = ['temperature', 'relative_humidity', 'rain', 'wind_speed10', 'wind_direction10']


def main():
    df = pd.read_csv(CSV_FP, index_col=0)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    hours = pd.date_range(WINDOW_START, WINDOW_END, freq='h')
    sub = df[(df.timestamp >= WINDOW_START) & (df.timestamp <= WINDOW_END)]
    sub = sub.drop_duplicates(['site_id', 'timestamp'])

    counts = sub.groupby('site_id').size()
    sites = sorted(counts[counts >= COVERAGE_THRESHOLD * len(hours)].index)
    print(f'{len(sites)} sites (of {df.site_id.nunique()} total) with >= {COVERAGE_THRESHOLD:.0%} '
          f'hourly coverage in {WINDOW_START} .. {WINDOW_END} ({len(hours)} hours)')

    cols = FEATURE_COLS + ['pm']
    arr = np.full((len(hours), len(sites), len(cols)), np.nan, dtype=np.float64)
    site_meta = {}
    n_interpolated = 0
    for j, site_id in enumerate(sites):
        site_rows = sub[sub.site_id == site_id]
        site_meta[site_id] = (site_rows['longitude'].iloc[0], site_rows['latitude'].iloc[0],
                               site_rows['elevation'].iloc[0])

        s = site_rows.set_index('timestamp')[cols].reindex(hours)
        n_missing_before = s.isna().any(axis=1).sum()
        # Linear time-interpolation fills the shared ~20h outage (and the
        # scattered ~0.5% raw missingness in the weather columns); ffill/
        # bfill mop up anything still missing at the very edges of the
        # window, where interpolation has no data on one side.
        s = s.interpolate(method='time').ffill().bfill()
        n_interpolated += n_missing_before
        arr[:, j, :] = s.to_numpy()

    assert not np.isnan(arr).any(), 'unfilled gaps remain - unexpected given the coverage filter above'
    pct_interp = 100 * n_interpolated / (len(hours) * len(sites))
    print(f'interpolated {n_interpolated} of {len(hours)*len(sites)} (site, hour) rows ({pct_interp:.1f}%)')

    np.save(NPY_FP, arr)
    print(f'wrote {NPY_FP}: shape={arr.shape} (time, node, feature) dtype={arr.dtype}')
    print(f'  features: {FEATURE_COLS} + [pm]')
    print(f'  time range: {hours[0]} .. {hours[-1]}, hourly')

    with open(SITE_FP, 'w') as f:
        for idx, site_id in enumerate(sites):
            lon, lat, elev = site_meta[site_id]
            f.write(f'{idx} {site_id} {lon} {lat} {elev}\n')
    print(f'wrote {SITE_FP}: {len(sites)} nodes')


if __name__ == '__main__':
    main()
