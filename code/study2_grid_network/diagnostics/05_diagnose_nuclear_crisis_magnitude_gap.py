# ═══════════════════════════════════════════════════════════════════════
# Resolve the nuclear crisis magnitude gap: proper same-season comparison,
# full-year sanity check, and per-country breakdown (is Italy the culprit?)
# ═══════════════════════════════════════════════════════════════════════

# %% [0] Setup ----------------------------------------------------------------
from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import numpy as np
import requests, zipfile, io, os, time, json

DATA_DIR = '/content/drive/MyDrive/grid_data'
os.chdir(DATA_DIR)
ZENODO_BASE = 'https://zenodo.org/records/13378476/files'
BASE_MVA = 100

with open('europe_network.json') as f:
    network = json.load(f)
with open('crossing_point_map.json') as f:
    crossing_point_map = json.load(f)
bus_country = {bus_id: info['country'] for bus_id, info in network['bus'].items()}

line_sign = {}
for name, line_ids in crossing_point_map.items():
    for lid in line_ids:
        info = network['branch'][lid]
        f_country = bus_country[str(info['f_bus'])]
        line_sign[lid] = 1 if f_country == 'FR' else -1

all_needed_line_ids = sorted(set(
    lid for line_ids in crossing_point_map.values() for lid in line_ids
))

# %% [1] Download 2018 (2016, 2017 already present — skip-if-exists) --------
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
                if downloaded_mb - last_reported_mb >= 20:
                    elapsed = time.time() - t0
                    rate = downloaded_mb / elapsed if elapsed > 0 else 0
                    if total_mb:
                        pct = 100 * downloaded_mb / total_mb
                        print(f'  {downloaded_mb:.0f} / {total_mb:.0f} MB ({pct:.0f}%)  {rate:.1f} MB/s', end='\r')
                    last_reported_mb = downloaded_mb
            print()
            print(f'  Download complete: {downloaded/(1024*1024):.0f} MB in {time.time()-t0:.1f}s')
            zip_bytes.seek(0)
            with zipfile.ZipFile(zip_bytes) as z:
                matching = [n for n in z.namelist() if member_pattern in n and '_1.csv' in n]
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
    raise ConnectionError(f'Failed to download {zip_name}')

YEARS = [2016, 2017, 2018]
for year in YEARS:
    download_and_extract_one(f'lines_{year}.zip', 'lines')

# %% [2] Rebuild sign-corrected series for all 3 years -----------------------
candidate_starts = {}
for year in [2016, 2017, 2018, 2019, 2020]:
    d = pd.Timestamp(f'{year}-01-01')
    days_to_monday = (7 - d.weekday()) % 7
    first_monday = d + pd.Timedelta(days=days_to_monday) if d.weekday() != 0 else d
    candidate_starts[year] = first_monday

crossing_series = {name: [] for name in crossing_point_map}
crossing_dates = {name: [] for name in crossing_point_map}

for year in YEARS:
    fname = f'lines_{year}_1.csv'
    print(f'Processing {fname}...')
    start_date = candidate_starts[year]
    date_range = pd.date_range(start_date, periods=8736, freq='h')
    header_cols = pd.read_csv(fname, nrows=0).columns
    cols_to_read = [c for c in all_needed_line_ids if c in header_cols]
    df_year = pd.read_csv(fname, usecols=cols_to_read, dtype=np.float32)
    for col in df_year.columns:
        if line_sign.get(col, 1) == -1:
            df_year[col] = -df_year[col]
    for name, line_ids in crossing_point_map.items():
        present = [lid for lid in line_ids if lid in df_year.columns]
        if not present:
            continue
        total_flow = df_year[present].sum(axis=1)
        crossing_series[name].extend(total_flow.tolist())
        crossing_dates[name].extend(date_range.tolist())
    del df_year

for name in crossing_point_map:
    df = pd.DataFrame({'Date': crossing_dates[name], 'Flow': crossing_series[name]}
                       ).sort_values('Date').drop_duplicates(subset='Date').reset_index(drop=True)
    df.to_csv(f'crossing_{name}.csv', index=False)

# Rebuild total + per-country aggregates
country_of = {}
for name in crossing_point_map:
    neighbour = name.split('_')[0].replace('FR-', '')
    country_of[name] = neighbour

total_export = None
per_country = {}
for name in crossing_point_map:
    df = pd.read_csv(f'crossing_{name}.csv', parse_dates=['Date']).set_index('Date')['Flow']
    total_export = df if total_export is None else total_export.add(df, fill_value=0)
    c = country_of[name]
    per_country[c] = df if c not in per_country else per_country[c].add(df, fill_value=0)

total_export_df = total_export.reset_index()
total_export_df.columns = ['Date', 'TotalNetExport']
total_export_df.to_csv('france_total_net_export.csv', index=False)

for c, series in per_country.items():
    df = series.reset_index()
    df.columns = ['Date', 'NetExport']
    df.to_csv(f'france_export_to_{c}.csv', index=False)

print(f'\nRebuilt total + {len(per_country)} per-country export series through {YEARS[-1]}.')

# %% [3] PROPER same-season comparison: Oct-Jan 2016/17 vs Oct-Jan 2017/18 --
df = pd.read_csv('france_total_net_export.csv', parse_dates=['Date'])
df['ExportGW'] = df['TotalNetExport'] * BASE_MVA / 1000

crisis = df[(df.Date >= '2016-10-10') & (df.Date <= '2017-01-27')]['ExportGW'].mean()
same_season_next_year = df[(df.Date >= '2017-10-10') & (df.Date <= '2018-01-27')]['ExportGW'].mean()

print('\n' + '=' * 70)
print('SAME-SEASON COMPARISON (controls for winter seasonal demand)')
print('=' * 70)
print(f'Crisis window (Oct 2016 - Jan 2017):        {crisis:+.2f} GW')
print(f'Same calendar window, following year (2017-18): {same_season_next_year:+.2f} GW')
print(f'Difference: {crisis - same_season_next_year:+.2f} GW')
print(f'(Real-world reported figure: approximately -7.2 GW)')

# %% [4] Full-year sanity check — does France net out positive over 2017? --
full_2017 = df[(df.Date >= '2017-01-01') & (df.Date < '2018-01-01')]['ExportGW'].mean()
print('\n' + '=' * 70)
print('FULL-YEAR SANITY CHECK (2017 — a year mostly clear of both events)')
print('=' * 70)
print(f'Mean net export, full year 2017: {full_2017:+.2f} GW')
print('(France is widely reported as a large net EXPORTER on average —')
print(' a negative full-year average would suggest a remaining issue.)')

# %% [5] Per-country breakdown — is Italy (largest group) the outlier? ------
print('\n' + '=' * 70)
print('PER-COUNTRY BREAKDOWN, full year 2017')
print('=' * 70)
for c in per_country:
    cdf = pd.read_csv(f'france_export_to_{c}.csv', parse_dates=['Date'])
    cdf['GW'] = cdf['NetExport'] * BASE_MVA / 1000
    yearly = cdf[(cdf.Date >= '2017-01-01') & (cdf.Date < '2018-01-01')]['GW'].mean()
    n_lines = len([l for l, cn in country_of.items() if cn == c])
    print(f'  France -> {c}: {yearly:+.2f} GW average (from {n_lines} crossing point(s))')

print('\nKnown reference (approximate real-world averages, for comparison):')
print('  France -> Italy: strongly positive (France is a major net exporter to Italy)')
print('  France -> UK: N/A (not in this continental-only model)')
print('  France -> others: smaller magnitudes, sign varies by year/season')
