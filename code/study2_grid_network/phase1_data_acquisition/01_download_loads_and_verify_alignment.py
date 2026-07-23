# ═══════════════════════════════════════════════════════════════════════
# Grid Anomaly Detection — Phase 1: Download, Aggregate, VERIFY DATE ALIGNMENT
# Run in Google Colab. Do NOT proceed to modelling until Phase 1 diagnostics
# confirm exact calendar alignment of each yearly file.
#
# Dataset: Gillioz et al. (2024), "A large synthetic dataset for machine
# learning applications in power transmission grids", Zenodo/Nature Sci Data
# DOI: 10.5281/zenodo.13378476
# ═══════════════════════════════════════════════════════════════════════

# %% [1] Imports and setup --------------------------------------------------
import pandas as pd
import numpy as np
import requests
import zipfile
import io
import os
import time

os.makedirs('/content/grid_data', exist_ok=True)
os.chdir('/content/grid_data')

ZENODO_BASE = 'https://zenodo.org/records/13378476/files'

# %% [2] Download the lightweight reference files first ---------------------
# These are small (KB-scale) and needed before touching the big zips

def download_file(filename, retries=5, initial_delay=5, backoff_factor=2, timeout=120):
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

download_file('loads_by_country.csv')
download_file('gens_by_country.csv')

loads_by_country = pd.read_csv('loads_by_country.csv', dtype=str)
gens_by_country = pd.read_csv('gens_by_country.csv', dtype=str)

print('\nCountries available in loads_by_country.csv:')
print(list(loads_by_country.columns))

# %% [3] Download ONLY index-1 CSVs from each year's zip (not all 4 variants)
# Each yearly zip contains 4 index variants (_1 to _4); we only need one
# per year to keep this tractable. This is still substantial (~1.3GB/year
# for loads, ~2.1GB/year for lines) so only download what's scoped.

YEARS = [2016]  # start with ONE year to verify the pipeline works and is fast
                 # enough before committing to all 5 — expand back to
                 # [2016, 2017, 2018, 2019, 2020] once this completes quickly

def download_and_extract_one(zip_name, member_pattern, retries=5, initial_delay=5,
                              backoff_factor=2, timeout=600):
    """Download a yearly zip WITH live progress + retries, extract only the
    _1 variant. Prints downloaded MB every ~10MB so a stall is never silent."""
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

            for chunk in r.iter_content(chunk_size=1024*1024):  # 1MB chunks
                zip_bytes.write(chunk)
                downloaded += len(chunk)
                downloaded_mb = downloaded / (1024*1024)
                if downloaded_mb - last_reported_mb >= 10:  # print every ~10MB
                    elapsed = time.time() - t0
                    rate = downloaded_mb / elapsed if elapsed > 0 else 0
                    if total_mb:
                        pct = 100 * downloaded_mb / total_mb
                        print(f'  {downloaded_mb:.0f} / {total_mb:.0f} MB '
                              f'({pct:.0f}%)  {rate:.1f} MB/s', end='\r')
                    else:
                        print(f'  {downloaded_mb:.0f} MB downloaded  '
                              f'{rate:.1f} MB/s', end='\r')
                    last_reported_mb = downloaded_mb

            print()  # newline after progress line
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

# Loads — needed for national aggregate demand
for year in YEARS:
    download_and_extract_one(f'loads_{year}.zip', 'loads')

# NOTE: lines_*.zip files are ~2.1GB EACH — only download these once Phase 1
# date alignment is confirmed and you've decided the France interconnector
# analysis is worth the download time/disk space. Commented out for now:
#
# for year in YEARS:
#     download_and_extract_one(f'lines_{year}.zip', 'lines')

# %% [4] CRITICAL DIAGNOSTIC — determine actual calendar alignment ----------
# The dataset documentation states each file is "a 364-day year... starting
# on a Monday" but does not give the exact start date per labelled year.
# We check this against a real, independently-known fact: 4 January 2016
# was a Monday, and the dataset is stated to be "statistically validated
# against real-world data" from ENTSO-E for the corresponding real years.
# This diagnostic does NOT assume the alignment — it lays out the candidate
# start dates for manual verification against known real events before
# any ground-truth labelling is trusted.

print('=' * 70)
print('DATE ALIGNMENT DIAGNOSTIC')
print('=' * 70)

# Candidate Monday start dates for each labelled year (2016-2020),
# taking the first Monday on or after 1 January of that year
candidate_starts = {}
for year in YEARS:
    d = pd.Timestamp(f'{year}-01-01')
    days_to_monday = (7 - d.weekday()) % 7
    first_monday = d + pd.Timedelta(days=days_to_monday) if d.weekday() != 0 else d
    candidate_starts[year] = first_monday

print('\nCandidate start dates (first Monday of each labelled year):')
for year, start in candidate_starts.items():
    end = start + pd.Timedelta(hours=8735)  # 8736 hours = 364*24
    print(f'  {year}: {start.date()} to {end.date()}  ({(end-start).days} days)')

print('\n⚠️  THIS IS A CANDIDATE ALIGNMENT, NOT A CONFIRMED ONE.')
print('Before trusting any ground-truth event label against this data:')
print('1. Load a load/gen column for a country with a well-known, sharp,')
print('   independently-dated real event NOT already used as ground truth')
print('   (e.g. a known public holiday demand dip) and check it lines up')
print('   with the candidate date above.')
print('2. Cross-check the francophone/DE consumption profile weekday vs')
print('   weekend pattern in the file against which day-of-week the')
print('   candidate start actually implies.')
print('3. Only once (1) or (2) confirms alignment, proceed to Phase 2')
print('   (ground-truth labelling and modelling).')

# %% [5] Build national aggregate load series (once alignment confirmed) ---
# This constructs the tractable, portfolio-scale time series: total demand
# per country, per labelled year, concatenated using the CANDIDATE dates
# above. Re-run this after alignment is confirmed/corrected in step 4.
#
# FIXED: previously re-read the full yearly CSV once per country (5x
# redundant disk reads per year). Now reads each yearly file ONCE, with
# only the union of needed columns across all target countries, then
# slices per-country from the in-memory DataFrame.

COUNTRIES = ['FR', 'DE', 'CH', 'IT', 'ES']  # adjust based on what's available

def get_country_load_list(country_code):
    return loads_by_country[country_code].dropna().tolist()

# Pre-compute each country's column list once, and the union across all
country_cols = {c: get_country_load_list(c) for c in COUNTRIES}
union_cols = sorted(set().union(*country_cols.values()))

country_series = {c: [] for c in COUNTRIES}
country_dates = {c: [] for c in COUNTRIES}

for year in YEARS:
    fname = f'loads_{year}_1.csv'
    if not os.path.exists(fname):
        print(f'Skipping {year} — file not downloaded (see step 3)')
        continue
    print(f'Processing {fname}...')
    start_date = candidate_starts[year]
    date_range = pd.date_range(start_date, periods=8736, freq='h')

    # Read the file's header ONCE to find which union columns actually exist
    header_cols = pd.read_csv(fname, nrows=0).columns
    cols_to_read = [c for c in union_cols if c in header_cols]

    # Read the ENTIRE year's data ONCE, only the columns we actually need.
    # dtype=np.float32 specified explicitly to skip pandas' per-column
    # dtype-inference overhead, which is a common slow-path on very wide
    # files (this file has 4097 total columns even though we only keep
    # a subset via usecols).
    print(f'  Reading {len(cols_to_read)} columns from {fname} (this may take a minute)...')
    import time as _time
    _t0 = _time.time()
    df_year = pd.read_csv(fname, usecols=cols_to_read, dtype=np.float32)
    print(f'  Loaded {df_year.shape[0]} rows x {df_year.shape[1]} cols '
          f'in {_time.time() - _t0:.1f} seconds')

    # Now slice per-country from the already-loaded DataFrame (fast, in-memory)
    for country in COUNTRIES:
        cols_present = [c for c in country_cols[country] if c in df_year.columns]
        if not cols_present:
            continue
        total_load = df_year[cols_present].sum(axis=1)
        country_series[country].extend(total_load.tolist())
        country_dates[country].extend(date_range.tolist())

    del df_year  # free memory before next year

# Build final per-country DataFrames
country_dfs = {}
for country in COUNTRIES:
    if len(country_series[country]) == 0:
        continue
    df = pd.DataFrame({
        'Date': country_dates[country],
        'TotalLoad': country_series[country]
    }).sort_values('Date').reset_index(drop=True)
    country_dfs[country] = df
    print(f'{country}: {len(df)} rows, {df.Date.min()} to {df.Date.max()}')
    df.to_csv(f'national_load_{country}.csv', index=False)

print('\nPhase 1 complete. National aggregate load series saved per country.')
print('DO NOT proceed to anomaly detection modelling until date alignment')
print('(step 4) has been independently verified.')
