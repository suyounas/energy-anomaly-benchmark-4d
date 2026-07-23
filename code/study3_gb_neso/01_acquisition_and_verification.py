# ═══════════════════════════════════════════════════════════════════════
# STUDY 3 — Phase 1: Acquire REAL GB national demand data from NESO
# (National Energy System Operator, formerly National Grid ESO).
#
# This is the first fully-REAL grid dataset in this research programme:
# Study 1 = real commodity prices; Study 2 = validated synthetic grid;
# Study 3 = real GB grid demand. Removes the "synthetic" qualifier.
#
# Ground-truth events (real, independently documented):
#   - COVID-19 first UK lockdown: 2020-03-23 to 2020-06-01
#   - Dec 2022 cold snap / first Demand Flexibility Service era:
#     2022-12-01 to 2022-12-17
#
# Design principles (same discipline as Studies 1-2):
#   - Discover files via NESO's CKAN API, don't hardcode resource GUIDs
#   - Verify the data empirically BEFORE modelling (weekday/weekend
#     ratio, Christmas dip, COVID magnitude vs. real-world reporting)
#   - STOP after acquisition + verification. Modelling is Phase 2/3.
#   - No result-rigging: the checks pass or fail on what the data shows.
# ═══════════════════════════════════════════════════════════════════════

from google.colab import drive
drive.mount('/content/drive')

import os
import io
import requests
import pandas as pd
import numpy as np

DATA_DIR = '/content/drive/MyDrive/neso_data'
os.makedirs(DATA_DIR, exist_ok=True)
os.chdir(DATA_DIR)

YEARS = list(range(2016, 2024))  # 2016-2023 inclusive
CKAN_API = 'https://api.neso.energy/api/3/action'

# %% [1] Discover the Historic Demand Data resources via CKAN API ----------
# NESO's data portal is CKAN-based. The historic demand dataset publishes
# one CSV per year, named like 'demanddata_2020'. We search for the
# package and map year -> resource download URL programmatically.

print('=' * 70)
print('STEP 1: Discovering demand data files via NESO CKAN API')
print('=' * 70)

resp = requests.get(f'{CKAN_API}/package_search',
                    params={'q': 'historic demand data', 'rows': 10},
                    timeout=60)
resp.raise_for_status()
results = resp.json()['result']['results']

year_to_url = {}
for pkg in results:
    for res in pkg.get('resources', []):
        name = (res.get('name') or '').lower()
        url = res.get('url') or ''
        fmt = (res.get('format') or '').upper()
        for year in YEARS:
            if str(year) in name and 'demand' in name and fmt == 'CSV':
                # Prefer exact 'demanddata_YYYY' style names over summaries
                if year not in year_to_url or 'demanddata' in name:
                    year_to_url[year] = url

print(f'Found resources for years: {sorted(year_to_url.keys())}')
missing = [y for y in YEARS if y not in year_to_url]
if missing:
    print(f'⚠️ Not found via API search: {missing}')
    print('If any year is missing, check https://www.neso.energy/data-portal')
    print('manually — the dataset name may have changed. Proceeding with')
    print('what was found.')

# %% [2] Download each year, parse half-hourly settlement periods ----------
# Format: SETTLEMENT_DATE + SETTLEMENT_PERIOD (1-48, each = 30 min),
# ND = National Demand (MW). We convert to a proper datetime index.

print('\n' + '=' * 70)
print('STEP 2: Downloading and parsing')
print('=' * 70)

def parse_year_csv(raw_bytes, year):
    df = pd.read_csv(io.BytesIO(raw_bytes))
    df.columns = [c.strip().upper() for c in df.columns]
    date_col = 'SETTLEMENT_DATE'
    period_col = 'SETTLEMENT_PERIOD'
    demand_col = 'ND'  # National Demand
    for required in [date_col, period_col, demand_col]:
        if required not in df.columns:
            raise ValueError(f'{year}: expected column {required} not found. '
                             f'Columns present: {list(df.columns)[:10]}')
    df[date_col] = pd.to_datetime(df[date_col], dayfirst=True,
                                   format='mixed', errors='coerce')
    df = df.dropna(subset=[date_col])
    # settlement period N covers [ (N-1)*30min , N*30min ) from midnight
    df['Datetime'] = df[date_col] + pd.to_timedelta(
        (df[period_col].astype(int) - 1) * 30, unit='m')
    out = df[['Datetime', demand_col]].rename(columns={demand_col: 'ND_MW'})
    out = out.sort_values('Datetime').reset_index(drop=True)
    return out

all_years = []
for year in sorted(year_to_url.keys()):
    fname = f'neso_demand_{year}.csv'
    if os.path.exists(fname):
        print(f'{year}: already on Drive, loading cached copy')
        all_years.append(pd.read_csv(fname, parse_dates=['Datetime']))
        continue
    url = year_to_url[year]
    print(f'{year}: downloading from {url[:80]}...')
    r = requests.get(url, timeout=300)
    r.raise_for_status()
    parsed = parse_year_csv(r.content, year)
    parsed.to_csv(fname, index=False)
    print(f'  -> {len(parsed)} half-hourly rows, saved {fname}')
    all_years.append(parsed)

full = pd.concat(all_years, ignore_index=True).sort_values('Datetime')
full = full.drop_duplicates(subset='Datetime').reset_index(drop=True)
full.to_csv('neso_demand_full.csv', index=False)
print(f'\nCombined: {len(full)} rows, '
      f'{full.Datetime.min()} to {full.Datetime.max()}')

# %% [3] VERIFICATION — trust nothing until checked ------------------------
print('\n' + '=' * 70)
print('STEP 3: Empirical verification (same checks as Study 2)')
print('=' * 70)

full['date'] = full.Datetime.dt.date
daily = full.groupby('date')['ND_MW'].mean().reset_index()
daily['date'] = pd.to_datetime(daily['date'])
daily['dow'] = daily['date'].dt.dayofweek

# Check A: weekday/weekend ratio (should be >1, typically 1.05-1.15 for GB)
wd = daily[daily.dow < 5]['ND_MW'].mean()
we = daily[daily.dow >= 5]['ND_MW'].mean()
print(f'\nA. Weekday/weekend demand ratio: {wd/we:.3f} '
      f'({"✅ plausible" if 1.02 < wd/we < 1.3 else "⚠️ unexpected — investigate"})')

# Check B: Christmas Day dip (real data, should be a large minimum)
for year in [2018, 2019, 2021]:
    xmas = daily[daily['date'] == f'{year}-12-25']['ND_MW']
    dec = daily[(daily['date'] >= f'{year}-12-01') &
                (daily['date'] <= f'{year}-12-20')]['ND_MW'].mean()
    if len(xmas):
        dip = (xmas.iloc[0] - dec) / dec * 100
        print(f'B. Christmas {year} vs early-Dec mean: {dip:+.1f}% '
              f'({"✅" if dip < -10 else "⚠️"})')

# Check C: COVID lockdown magnitude vs real-world reporting (~10-20% GB)
covid_window = daily[(daily['date'] >= '2020-03-23') &
                     (daily['date'] <= '2020-05-11')]['ND_MW'].mean()
baseline = daily[(daily['date'] >= '2020-01-06') &
                 (daily['date'] <= '2020-03-08')]['ND_MW'].mean()
covid_drop = (covid_window - baseline) / baseline * 100
print(f'C. COVID lockdown demand change vs pre-lockdown 2020: '
      f'{covid_drop:+.1f}%')
print(f'   Real-world GB reporting was roughly -10 to -20%. '
      f'({"✅ consistent" if -25 < covid_drop < -8 else "⚠️ outside expected range"})')
print('   NOTE: seasonal demand decline (winter->spring) inflates this;')
print('   Phase 2 should use a year-over-year comparison for the paper.')

# Check D: Dec 2022 cold snap (demand should SPIKE vs early Dec)
cold = daily[(daily['date'] >= '2022-12-08') &
             (daily['date'] <= '2022-12-17')]['ND_MW'].mean()
early_dec = daily[(daily['date'] >= '2022-11-20') &
                  (daily['date'] <= '2022-12-05')]['ND_MW'].mean()
cold_change = (cold - early_dec) / early_dec * 100
print(f'D. Dec 2022 cold snap vs late Nov/early Dec: {cold_change:+.1f}% '
      f'({"✅ elevated as expected" if cold_change > 3 else "⚠️ weaker than expected"})')

# Check E: gaps
full_sorted = full.sort_values('Datetime')
diffs = full_sorted.Datetime.diff().dropna()
gaps = diffs[diffs > pd.Timedelta('30min')]
print(f'\nE. Gaps larger than one settlement period: {len(gaps)}')
if len(gaps):
    print('   Largest 5:')
    for idx in gaps.nlargest(5).index:
        print(f'   {full_sorted.Datetime[idx - 1]} -> '
              f'{full_sorted.Datetime[idx]} ({diffs[idx]})')

print('\n' + '=' * 70)
print('PHASE 1 COMPLETE — data acquired and verified.')
print('STOP HERE. Review the checks above before any modelling.')
print('If A-E look right, Phase 2 = ground truth labelling + features,')
print('reusing the Study 2 recipe unchanged (no method changes = the')
print('cross-dataset comparison stays clean).')
print('=' * 70)
