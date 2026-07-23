# ═══════════════════════════════════════════════════════════════════════
# Phase 2a (SELF-CONTAINED, Drive-persistent version)
# Safe to run in a fresh runtime — defines everything itself, and SKIPS
# re-downloading any file that already exists (in Drive or locally).
# ═══════════════════════════════════════════════════════════════════════

# %% [0] Mount Drive for persistence across disconnects/restarts -----------
from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import numpy as np
import requests
import zipfile
import io
import os
import time

DATA_DIR = '/content/drive/MyDrive/grid_data'
os.makedirs(DATA_DIR, exist_ok=True)
os.chdir(DATA_DIR)
print(f'Working directory: {os.getcwd()}')
print(f'Files already present: {os.listdir(".")}')

ZENODO_BASE = 'https://zenodo.org/records/13378476/files'

# %% [1] Function definitions (self-contained, no dependency on Phase 1) ---

def download_file(filename, retries=5, initial_delay=5, backoff_factor=2, timeout=120):
    if os.path.exists(filename):
        print(f'{filename} already exists — skipping download.')
        return
    url = f'{ZENODO_BASE}/{filename}?download=1'
    for attempt in range(retries):
        try:
            print(f'Downloading {filename} (attempt {attempt+1}/{retries})...')
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            with open(filename, 'wb') as f:
                f.write(r.content)
            print(f'  -> {len(r.content)/1024:.1f} KB saved')
            return
        except requests.exceptions.RequestException as e:
            delay = initial_delay * (backoff_factor ** attempt)
            print(f'  Error: {e}. Retrying in {delay:.1f}s...')
            time.sleep(delay)
    raise ConnectionError(f'Failed to download {filename} after {retries} attempts.')


def download_and_extract_one(zip_name, member_pattern, retries=5, initial_delay=5,
                              backoff_factor=2, timeout=600):
    """Download a yearly zip WITH live progress + retries, extract only the
    _1 variant. SKIPS entirely if the extracted CSV already exists."""
    expected_csv = zip_name.replace('.zip', '_1.csv')
    if os.path.exists(expected_csv):
        print(f'{expected_csv} already exists — skipping {zip_name} download entirely.')
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
                if downloaded_mb - last_reported_mb >= 10:
                    elapsed = time.time() - t0
                    rate = downloaded_mb / elapsed if elapsed > 0 else 0
                    if total_mb:
                        pct = 100 * downloaded_mb / total_mb
                        print(f'  {downloaded_mb:.0f} / {total_mb:.0f} MB '
                              f'({pct:.0f}%)  {rate:.1f} MB/s', end='\r')
                    else:
                        print(f'  {downloaded_mb:.0f} MB downloaded  {rate:.1f} MB/s', end='\r')
                    last_reported_mb = downloaded_mb

            print()
            print(f'  Download complete: {downloaded/(1024*1024):.0f} MB in '
                  f'{time.time()-t0:.1f}s')
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
    raise ConnectionError(f'Failed to download and extract {zip_name} after {retries} attempts.')

# %% [2] Reference files ----------------------------------------------------
download_file('loads_by_country.csv')
download_file('gens_by_country.csv')
loads_by_country = pd.read_csv('loads_by_country.csv', dtype=str)

# %% [3] Download all 5 years (each SKIPPED automatically if already present)
ALL_YEARS = [2016, 2017, 2018, 2019, 2020]
for year in ALL_YEARS:
    download_and_extract_one(f'loads_{year}.zip', 'loads')

# %% [4] Candidate start dates ----------------------------------------------
candidate_starts = {}
for year in ALL_YEARS:
    d = pd.Timestamp(f'{year}-01-01')
    days_to_monday = (7 - d.weekday()) % 7
    first_monday = d + pd.Timedelta(days=days_to_monday) if d.weekday() != 0 else d
    candidate_starts[year] = first_monday

print('\nCandidate start dates:')
for year, start in candidate_starts.items():
    print(f'  {year}: {start.date()}')

# %% [5] Build full 5-year series, all countries ----------------------------
ALL_COUNTRIES = list(loads_by_country.columns)
country_cols = {c: loads_by_country[c].dropna().tolist() for c in ALL_COUNTRIES}
union_cols = sorted(set().union(*country_cols.values()))

country_series = {c: [] for c in ALL_COUNTRIES}
country_dates = {c: [] for c in ALL_COUNTRIES}

for year in ALL_YEARS:
    fname = f'loads_{year}_1.csv'
    if not os.path.exists(fname):
        print(f'Skipping {year} — file not found')
        continue
    print(f'Processing {fname}...')
    start_date = candidate_starts[year]
    date_range = pd.date_range(start_date, periods=8736, freq='h')

    header_cols = pd.read_csv(fname, nrows=0).columns
    cols_to_read = [c for c in union_cols if c in header_cols]

    df_year = pd.read_csv(fname, usecols=cols_to_read, dtype=np.float32)
    print(f'  Loaded {df_year.shape[0]} x {df_year.shape[1]}')

    for country in ALL_COUNTRIES:
        cols_present = [c for c in country_cols[country] if c in df_year.columns]
        if not cols_present:
            continue
        total_load = df_year[cols_present].sum(axis=1)
        country_series[country].extend(total_load.tolist())
        country_dates[country].extend(date_range.tolist())

    del df_year

# %% [6] Year-boundary continuity check (critical for nuclear crisis event)
print('\n' + '=' * 70)
print('YEAR BOUNDARY CONTINUITY CHECK')
print('=' * 70)

df_fr = pd.DataFrame({
    'Date': country_dates['FR'],
    'TotalLoad': country_series['FR']
}).sort_values('Date').reset_index(drop=True)

diffs = df_fr['Date'].diff().dropna()
gaps = diffs[diffs != pd.Timedelta(hours=1)]
print(f'Total rows: {len(df_fr)}  |  Non-1-hour gaps: {len(gaps)}')
if len(gaps) > 0:
    for idx in gaps.index[:10]:
        print(f'  Gap: {df_fr.loc[idx-1,"Date"]} -> {df_fr.loc[idx,"Date"]} = {diffs.loc[idx]}')
else:
    print('✅ No gaps.')
duplicates = df_fr['Date'].duplicated().sum()
print(f'Duplicate timestamps: {duplicates}' + (' ✅' if duplicates == 0 else ' ⚠️'))

boundary = df_fr[(df_fr.Date >= '2016-12-30') & (df_fr.Date <= '2017-01-03')]
print(f'\n2016->2017 boundary ({len(boundary)} rows):')
print(boundary.to_string(index=False))

# %% [7] Save full per-country series ---------------------------------------
country_dfs = {}
for country in ALL_COUNTRIES:
    if len(country_series[country]) == 0:
        continue
    df = pd.DataFrame({
        'Date': country_dates[country],
        'TotalLoad': country_series[country]
    }).sort_values('Date').drop_duplicates(subset='Date').reset_index(drop=True)
    country_dfs[country] = df
    df.to_csv(f'national_load_{country}_full.csv', index=False)

print(f'\nSaved {len(country_dfs)} country series to {DATA_DIR}')
print('These are now on Google Drive — safe across disconnects/restarts.')
print('\nPhase 2a (self-contained) complete. Review the boundary check above')
print('before proceeding to ground truth + features + modelling.')
