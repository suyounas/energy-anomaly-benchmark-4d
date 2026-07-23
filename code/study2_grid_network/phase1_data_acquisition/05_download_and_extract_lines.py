# ═══════════════════════════════════════════════════════════════════════
# Line flow download + extraction — France-neighbour interconnectors only
# Self-contained: mounts Drive, defines its own functions, skips anything
# already downloaded. Scoped to 2016-2017 first (sufficient for the nuclear
# crisis event) — see bottom of file for how to expand to all 5 years.
# ═══════════════════════════════════════════════════════════════════════

# %% [0] Setup ---------------------------------------------------------------
from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import numpy as np
import requests
import zipfile
import io
import os
import time
import json

DATA_DIR = '/content/drive/MyDrive/grid_data'
os.chdir(DATA_DIR)
ZENODO_BASE = 'https://zenodo.org/records/13378476/files'

# %% [1] Load the crossing point map from the previous step -----------------
with open('crossing_point_map.json') as f:
    crossing_point_map = json.load(f)

# Flatten to the full list of individual line-ID columns we need to extract
all_needed_line_ids = sorted(set(
    lid for line_ids in crossing_point_map.values() for lid in line_ids
))
print(f'Crossing points: {len(crossing_point_map)}')
print(f'Individual line columns needed: {len(all_needed_line_ids)}')

# %% [2] Download function (same pattern as loads — retries + live progress)
def download_and_extract_one(zip_name, member_pattern, retries=5, initial_delay=5,
                              backoff_factor=2, timeout=600):
    expected_csv = zip_name.replace('.zip', '_1.csv')
    if os.path.exists(expected_csv):
        print(f'{expected_csv} already exists — skipping {zip_name} download.')
        return

    url = f'{ZENODO_BASE}/{zip_name}?download=1'
    for attempt in range(retries):
        try:
            print(f'Downloading {zip_name} (attempt {attempt+1}/{retries})...')
            r = requests.get(url, timeout=timeout, stream=True)
            r.raise_for_status()
            total_bytes = int(r.headers.get('content-length', 0))
            total_mb = total_bytes / (1024*1024) if total_bytes else None

            zip_bytes = io.BytesIO()
            downloaded = 0
            last_reported_mb = 0
            t0 = time.time()

            for chunk in r.iter_content(chunk_size=1024*1024):
                zip_bytes.write(chunk)
                downloaded += len(chunk)
                downloaded_mb = downloaded / (1024*1024)
                if downloaded_mb - last_reported_mb >= 20:  # every 20MB given larger files
                    elapsed = time.time() - t0
                    rate = downloaded_mb / elapsed if elapsed > 0 else 0
                    if total_mb:
                        pct = 100 * downloaded_mb / total_mb
                        print(f'  {downloaded_mb:.0f} / {total_mb:.0f} MB '
                              f'({pct:.0f}%)  {rate:.1f} MB/s', end='\r')
                    else:
                        print(f'  {downloaded_mb:.0f} MB  {rate:.1f} MB/s', end='\r')
                    last_reported_mb = downloaded_mb

            print()
            print(f'  Download complete: {downloaded/(1024*1024):.0f} MB in {time.time()-t0:.1f}s')
            zip_bytes.seek(0)

            with zipfile.ZipFile(zip_bytes) as z:
                matching = [n for n in z.namelist() if member_pattern in n and '_1.csv' in n]
                print(f'  Found matching members: {matching}')
                for m in matching:
                    z.extract(m, '.')
                    print(f'  Extracted: {m}')
            del zip_bytes
            print(f'  Done with {zip_name}\n')
            return
        except requests.exceptions.RequestException as e:
            delay = initial_delay * (backoff_factor ** attempt)
            print(f'\n  Error: {e}. Retrying in {delay:.1f}s...')
            time.sleep(delay)
    raise ConnectionError(f'Failed to download {zip_name} after {retries} attempts.')

# %% [3] Download the years we need (START with just 2016-2017) -------------
YEARS = [2016, 2017]  # expand to [2016, 2017, 2018, 2019, 2020] once verified

for year in YEARS:
    download_and_extract_one(f'lines_{year}.zip', 'lines')

# %% [4] Candidate start dates (identical logic/values as verified for loads)
candidate_starts = {}
for year in [2016, 2017, 2018, 2019, 2020]:
    d = pd.Timestamp(f'{year}-01-01')
    days_to_monday = (7 - d.weekday()) % 7
    first_monday = d + pd.Timedelta(days=days_to_monday) if d.weekday() != 0 else d
    candidate_starts[year] = first_monday

# %% [5] Extract only the needed columns, aggregate into crossing points ----
crossing_series = {name: [] for name in crossing_point_map}
crossing_dates = {name: [] for name in crossing_point_map}

for year in YEARS:
    fname = f'lines_{year}_1.csv'
    if not os.path.exists(fname):
        print(f'Skipping {year} — file not found')
        continue
    print(f'Processing {fname}...')
    start_date = candidate_starts[year]
    date_range = pd.date_range(start_date, periods=8736, freq='h')

    header_cols = pd.read_csv(fname, nrows=0).columns
    cols_to_read = [c for c in all_needed_line_ids if c in header_cols]
    missing = set(all_needed_line_ids) - set(cols_to_read)
    if missing:
        print(f'  ⚠️ {len(missing)} expected line columns not found in {fname}: {missing}')

    print(f'  Reading {len(cols_to_read)} of {len(header_cols)} total columns...')
    t0 = time.time()
    df_year = pd.read_csv(fname, usecols=cols_to_read, dtype=np.float32)
    print(f'  Loaded {df_year.shape[0]} x {df_year.shape[1]} in {time.time()-t0:.1f}s')

    for name, line_ids in crossing_point_map.items():
        present = [lid for lid in line_ids if lid in df_year.columns]
        if not present:
            continue
        # Line flows can be signed (direction matters) — sum preserves net
        # flow direction across parallel circuits, which is physically correct
        total_flow = df_year[present].sum(axis=1)
        crossing_series[name].extend(total_flow.tolist())
        crossing_dates[name].extend(date_range.tolist())

    del df_year

# %% [6] Boundary continuity check (same check as loads, applied to lines) --
print('\n' + '=' * 70)
print('YEAR BOUNDARY CONTINUITY CHECK (line flow data)')
print('=' * 70)
test_point = list(crossing_point_map.keys())[0]
df_test = pd.DataFrame({
    'Date': crossing_dates[test_point],
    'Flow': crossing_series[test_point]
}).sort_values('Date').reset_index(drop=True)

diffs = df_test['Date'].diff().dropna()
gaps = diffs[diffs != pd.Timedelta(hours=1)]
print(f'Test point: {test_point}')
print(f'Total rows: {len(df_test)}  |  Non-1-hour gaps: {len(gaps)}')
if len(gaps) > 0:
    for idx in gaps.index[:5]:
        print(f'  Gap: {df_test.loc[idx-1,"Date"]} -> {df_test.loc[idx,"Date"]}')
else:
    print('✅ No gaps in this range.')

# %% [7] Save all crossing-point series to Drive -----------------------------
for name in crossing_point_map:
    if len(crossing_series[name]) == 0:
        continue
    df = pd.DataFrame({
        'Date': crossing_dates[name],
        'Flow': crossing_series[name]
    }).sort_values('Date').drop_duplicates(subset='Date').reset_index(drop=True)
    df.to_csv(f'crossing_{name}.csv', index=False)

print(f'\nSaved {len(crossing_point_map)} crossing-point flow series to {DATA_DIR}')
print('\nTo expand to all 5 years later: change YEARS in step [3] to')
print('[2016, 2017, 2018, 2019, 2020] and re-run — already-downloaded')
print('years will be skipped automatically.')
