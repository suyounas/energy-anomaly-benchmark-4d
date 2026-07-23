# ═══════════════════════════════════════════════════════════════════════
# STUDY 4 — Phase 1: Acquire REAL US grid demand data (ERCOT, Texas)
# from EIA-930 Hourly Grid Monitor balance files. Direct CSV downloads,
# no API key required.
#
# Ground-truth events (real, independently documented):
#   - Texas Freeze / Winter Storm Uri: 2021-02-14 to 2021-02-19
#     ⚠ IMPORTANT: ERCOT shed up to ~20 GW of load, so RECORDED demand
#     UNDERSTATES true demand — the event manifests as demand spike THEN
#     suppressed/collapsed recorded demand. Same event Study 1 detected
#     in Henry Hub prices — cross-data-type closure.
#   - Winter Storm Elliott: 2022-12-22 to 2022-12-26 (demand spike,
#     no comparable load shed in ERCOT)
#
# EIA-930 coverage starts July 2018. Files are LARGE (all balancing
# authorities together), so we stream-filter each file to ERCO rows only
# and save just those — keeps Drive usage small.
#
# Same discipline: acquire, verify, STOP before modelling.
# ═══════════════════════════════════════════════════════════════════════

from google.colab import drive
drive.mount('/content/drive')

import os
import requests
import pandas as pd
import numpy as np

DATA_DIR = '/content/drive/MyDrive/eia_data'
os.makedirs(DATA_DIR, exist_ok=True)
os.chdir(DATA_DIR)

BASE = 'https://www.eia.gov/electricity/gridmonitor/sixMonthFiles'
HALVES = [(y, h) for y in range(2019, 2024) for h in ('Jan_Jun', 'Jul_Dec')]
BA = 'ERCO'  # ERCOT balancing authority code in EIA-930

# %% [1] Download + stream-filter each 6-month file to ERCOT only ----------
print('=' * 70)
print('STEP 1: Download 6-month EIA-930 balance files, filter to ERCOT')
print('=' * 70)

def fetch_and_filter(year, half):
    out_name = f'ercot_{year}_{half}.csv'
    if os.path.exists(out_name):
        print(f'{year} {half}: cached on Drive')
        return pd.read_csv(out_name)
    url = f'{BASE}/EIA930_BALANCE_{year}_{half}.csv'
    print(f'{year} {half}: downloading {url}')
    tmp = f'/content/tmp_{year}_{half}.csv'   # local disk, not Drive
    with requests.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(tmp, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    # Stream-read in chunks, keep only ERCOT rows
    kept = []
    for chunk in pd.read_csv(tmp, chunksize=200_000, low_memory=False):
        ba_col = next((c for c in chunk.columns if 'Balancing Authority' in c), None)
        if ba_col is None:
            raise ValueError(f'{year} {half}: no Balancing Authority column. '
                             f'Columns: {list(chunk.columns)[:8]}')
        kept.append(chunk[chunk[ba_col] == BA])
    os.remove(tmp)
    df = pd.concat(kept, ignore_index=True)
    df.to_csv(out_name, index=False)
    print(f'  -> {len(df)} ERCOT rows saved to {out_name}')
    return df

frames = []
for year, half in HALVES:
    try:
        frames.append(fetch_and_filter(year, half))
    except Exception as e:
        print(f'  ⚠️ {year} {half} failed: {e} — continuing')

raw = pd.concat(frames, ignore_index=True)
print(f'\nTotal ERCOT rows across all files: {len(raw)}')

# %% [2] Parse to a clean hourly demand series ------------------------------
print('\n' + '=' * 70)
print('STEP 2: Parsing to clean hourly series')
print('=' * 70)

cols = list(raw.columns)
# Prefer adjusted demand where present; fall back to raw demand
demand_col = next((c for c in cols if 'Demand (MW) (Adjusted)' in c), None) \
    or next((c for c in cols if c.strip() == 'Demand (MW)'), None) \
    or next((c for c in cols if 'Demand (MW)' in c and 'Forecast' not in c), None)
time_col = next((c for c in cols if 'UTC Time at End of Hour' in c), None) \
    or next((c for c in cols if 'Local Time at End of Hour' in c), None)
print(f'Using demand column: "{demand_col}"')
print(f'Using time column:   "{time_col}"')

df = raw[[time_col, demand_col]].copy()
df.columns = ['Datetime', 'Demand_MW']
df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')
# EIA numbers may carry thousands separators as strings
if df['Demand_MW'].dtype == object:
    df['Demand_MW'] = pd.to_numeric(
        df['Demand_MW'].astype(str).str.replace(',', ''), errors='coerce')
df = df.dropna().sort_values('Datetime').drop_duplicates('Datetime').reset_index(drop=True)
df.to_csv('ercot_demand_full.csv', index=False)
print(f'Clean series: {len(df)} hourly rows, {df.Datetime.min()} to {df.Datetime.max()}')

# %% [3] VERIFICATION -------------------------------------------------------
print('\n' + '=' * 70)
print('STEP 3: Empirical verification')
print('=' * 70)

df['date'] = df.Datetime.dt.date
daily = df.groupby('date')['Demand_MW'].mean().reset_index()
daily['date'] = pd.to_datetime(daily['date'])
daily['dow'] = daily['date'].dt.dayofweek

# A: weekday/weekend ratio (ERCOT has big AC load; ratio nearer 1.05)
wd = daily[daily.dow < 5]['Demand_MW'].mean()
we = daily[daily.dow >= 5]['Demand_MW'].mean()
print(f'\nA. Weekday/weekend ratio: {wd/we:.3f} '
      f'({"✅ plausible" if 1.0 < wd/we < 1.25 else "⚠️ investigate"})')

# B: Christmas dip
for year in [2019, 2021]:
    xmas = daily[daily['date'] == f'{year}-12-25']['Demand_MW']
    dec = daily[(daily['date'] >= f'{year}-12-01') &
                (daily['date'] <= f'{year}-12-20')]['Demand_MW'].mean()
    if len(xmas):
        dip = (xmas.iloc[0] - dec) / dec * 100
        print(f'B. Christmas {year} vs early-Dec mean: {dip:+.1f}% '
              f'({"✅" if dip < -3 else "⚠️ (US Xmas dip is milder than GB)"})')

# C: Texas Freeze signature — two-part check because of load shed.
#    Part 1: days BEFORE shed (Feb 10-14) should show elevated demand.
#    Part 2: YoY comparison over the full event window, acknowledging
#            recorded demand is suppressed by up to ~20 GW of shed load.
pre_freeze = daily[(daily['date'] >= '2021-02-10') &
                   (daily['date'] <= '2021-02-14')]['Demand_MW'].mean()
feb_baseline = np.nanmean([
    daily[(daily['date'] >= f'{y}-02-01') &
          (daily['date'] <= f'{y}-02-09')]['Demand_MW'].mean()
    for y in (2019, 2020, 2021)])
print(f'\nC1. Pre-shed cold ramp (10-14 Feb 2021) vs early-Feb baseline: '
      f'{(pre_freeze - feb_baseline)/feb_baseline*100:+.1f}% '
      f'({"✅ elevated" if pre_freeze > feb_baseline else "⚠️"})')

freeze = daily[(daily['date'] >= '2021-02-15') &
               (daily['date'] <= '2021-02-18')]['Demand_MW'].mean()
same_prior = np.nanmean([
    daily[(daily['date'] >= f'{y}-02-15') &
          (daily['date'] <= f'{y}-02-18')]['Demand_MW'].mean()
    for y in (2019, 2020)])
print(f'C2. Shed period (15-18 Feb 2021) vs same dates 2019/20: '
      f'{(freeze - same_prior)/same_prior*100:+.1f}%')
print('    Expected: ambiguous or NEGATIVE despite record cold, because')
print('    up to ~20 GW of demand was forcibly shed. Recorded demand is')
print('    supply-constrained here — document this, do not "fix" it.')

# D: Winter Storm Elliott (no major ERCOT shed → expect a genuine spike)
elliott = daily[(daily['date'] >= '2022-12-22') &
                (daily['date'] <= '2022-12-26')]['Demand_MW'].mean()
elliott_prior = np.nanmean([
    daily[(daily['date'] >= f'{y}-12-22') &
          (daily['date'] <= f'{y}-12-26')]['Demand_MW'].mean()
    for y in (2019, 2021)])
print(f'\nD. Elliott (22-26 Dec 2022) vs same dates 2019/21: '
      f'{(elliott - elliott_prior)/elliott_prior*100:+.1f}% '
      f'({"✅ spike as expected" if elliott > elliott_prior * 1.05 else "⚠️ weaker than expected"})')

# E: gaps
diffs = df.sort_values('Datetime').Datetime.diff().dropna()
gaps = diffs[diffs > pd.Timedelta('1h')]
print(f'\nE. Gaps larger than one hour: {len(gaps)}')
for idx in gaps.nlargest(5).index:
    print(f'   ...{diffs[idx]} ending {df.Datetime[idx]}')

print('\n' + '=' * 70)
print('PHASE 1 COMPLETE — STOP. Review A-E before modelling.')
print('Phase 2 will label: Freeze (2021-02-14 to 2021-02-19) and')
print('Elliott (2022-12-22 to 2022-12-26), then reuse the identical')
print('feature/split/modelling recipe. Note: data starts mid-2018, so')
print('there is NO COVID-clean long history — Protocol B window choice')
print('will differ from Study 3 and must be documented.')
print('=' * 70)
