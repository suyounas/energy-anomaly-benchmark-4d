# Run in the same Colab session. Re-aggregates the already-downloaded
# lines_2016_1.csv / lines_2017_1.csv with a CONSISTENT sign convention:
# positive = France exporting TO the neighbour, for every crossing point.
# No re-download needed — raw files are still on disk.

import json
import os
import pandas as pd
import numpy as np

DATA_DIR = '/content/drive/MyDrive/grid_data'
os.chdir(DATA_DIR)

with open('europe_network.json') as f:
    network = json.load(f)
with open('crossing_point_map.json') as f:
    crossing_point_map = json.load(f)

bus_country = {bus_id: info['country'] for bus_id, info in network['bus'].items()}

# %% Determine sign multiplier for EVERY individual line (not just per group)
# +1 if France is f_bus (positive = France exporting, already correct)
# -1 if France is t_bus (positive = neighbour exporting TO France, needs flip)
line_sign = {}
for name, line_ids in crossing_point_map.items():
    for lid in line_ids:
        info = network['branch'][lid]
        f_country = bus_country[str(info['f_bus'])]
        line_sign[lid] = 1 if f_country == 'FR' else -1

flipped = [lid for lid, s in line_sign.items() if s == -1]
print(f'Lines requiring sign flip (France is t_bus, not f_bus): {len(flipped)}')
print(f'Flipped line IDs: {flipped}')

# Sanity check: within a single crossing point, all parallel circuits should
# have the SAME orientation (they connect the identical two buses) — verify
# this rather than assume it
print('\nVerifying orientation is consistent WITHIN each crossing point...')
inconsistent = []
for name, line_ids in crossing_point_map.items():
    signs = set(line_sign[lid] for lid in line_ids)
    if len(signs) > 1:
        inconsistent.append(name)
if inconsistent:
    print(f'⚠️ INCONSISTENT orientation within: {inconsistent} — investigate before proceeding')
else:
    print('✅ Every crossing point has consistent orientation across its parallel circuits.')

# %% Re-read raw line data and re-aggregate WITH sign correction ------------
YEARS = [2016, 2017]

candidate_starts = {}
for year in [2016, 2017, 2018, 2019, 2020]:
    d = pd.Timestamp(f'{year}-01-01')
    days_to_monday = (7 - d.weekday()) % 7
    first_monday = d + pd.Timedelta(days=days_to_monday) if d.weekday() != 0 else d
    candidate_starts[year] = first_monday

all_needed_line_ids = sorted(set(
    lid for line_ids in crossing_point_map.values() for lid in line_ids
))

crossing_series = {name: [] for name in crossing_point_map}
crossing_dates = {name: [] for name in crossing_point_map}

for year in YEARS:
    fname = f'lines_{year}_1.csv'
    print(f'Re-processing {fname} with sign correction...')
    start_date = candidate_starts[year]
    date_range = pd.date_range(start_date, periods=8736, freq='h')

    header_cols = pd.read_csv(fname, nrows=0).columns
    cols_to_read = [c for c in all_needed_line_ids if c in header_cols]
    df_year = pd.read_csv(fname, usecols=cols_to_read, dtype=np.float32)

    # Apply sign correction BEFORE summing
    for col in df_year.columns:
        if line_sign.get(col, 1) == -1:
            df_year[col] = -df_year[col]

    for name, line_ids in crossing_point_map.items():
        present = [lid for lid in line_ids if lid in df_year.columns]
        if not present:
            continue
        total_flow = df_year[present].sum(axis=1)  # now consistently signed
        crossing_series[name].extend(total_flow.tolist())
        crossing_dates[name].extend(date_range.tolist())

    del df_year

# %% Save corrected series (overwrites the earlier uncorrected ones) --------
for name in crossing_point_map:
    if len(crossing_series[name]) == 0:
        continue
    df = pd.DataFrame({
        'Date': crossing_dates[name],
        'Flow': crossing_series[name]
    }).sort_values('Date').drop_duplicates(subset='Date').reset_index(drop=True)
    df.to_csv(f'crossing_{name}.csv', index=False)

print(f'\nSaved {len(crossing_point_map)} SIGN-CORRECTED crossing-point series.')
print('Positive value now consistently means: France exporting TO the neighbour,')
print('for every one of the 28 series.')

# %% Build an aggregate "total France net export" series across ALL points -
total_export = None
for name in crossing_point_map:
    df = pd.read_csv(f'crossing_{name}.csv', parse_dates=['Date'])
    df = df.set_index('Date')['Flow']
    total_export = df if total_export is None else total_export.add(df, fill_value=0)

total_export_df = total_export.reset_index()
total_export_df.columns = ['Date', 'TotalNetExport']
total_export_df.to_csv('france_total_net_export.csv', index=False)
print(f'\nSaved france_total_net_export.csv: {len(total_export_df)} rows')
print('This is now a physically meaningful single series: France\'s total net')
print('export across all interconnectors, correctly signed and ready for the')
print('nuclear crisis verification check.')
