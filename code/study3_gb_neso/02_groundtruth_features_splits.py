# ═══════════════════════════════════════════════════════════════════════
# STUDY 3 — Phase 2: Year-over-year event verification, ground-truth
# labelling, feature engineering, and BOTH split protocols (A and B),
# reusing the Study 2 recipe so the cross-dataset comparison stays clean.
#
# Only deliberate change from Study 2: data is HALF-HOURLY (48/day), so
# "30-day" rolling windows = 1440 rows instead of 720. Feature semantics
# are unchanged.
#
# STOPS before modelling, as always.
# ═══════════════════════════════════════════════════════════════════════

from google.colab import drive
drive.mount('/content/drive')

import os
import pandas as pd
import numpy as np

DATA_DIR = '/content/drive/MyDrive/neso_data'
os.chdir(DATA_DIR)

full = pd.read_csv('neso_demand_full.csv', parse_dates=['Datetime'])
full = full.sort_values('Datetime').reset_index(drop=True)
print(f'Loaded {len(full)} rows, {full.Datetime.min()} to {full.Datetime.max()}')

# ═══════════════════════════════════════════════════════════════════════
# %% [1] PAPER-GRADE COVID VERIFICATION: year-over-year, same calendar
# window, controlling for seasonality. This replaces the crude Phase 1
# check that conflated lockdown with the winter->spring decline.
# ═══════════════════════════════════════════════════════════════════════
print('\n' + '=' * 70)
print('STEP 1: Year-over-year COVID check (same calendar window)')
print('=' * 70)

def window_mean(df, year, start_md, end_md):
    lo = pd.Timestamp(f'{year}-{start_md}')
    hi = pd.Timestamp(f'{year}-{end_md}')
    sel = df[(df.Datetime >= lo) & (df.Datetime <= hi)]['ND_MW']
    return sel.mean() if len(sel) else np.nan

WINDOW = ('03-23', '05-31')  # lockdown calendar window
m2020 = window_mean(full, 2020, *WINDOW)
prior = [window_mean(full, y, *WINDOW) for y in (2017, 2018, 2019)]
prior_mean = np.nanmean(prior)

yoy = (m2020 - prior_mean) / prior_mean * 100
print(f'Mean demand 23 Mar–31 May 2020:        {m2020:,.0f} MW')
print(f'Same window, 2017-2019 average:        {prior_mean:,.0f} MW')
for y, v in zip((2017, 2018, 2019), prior):
    print(f'  ({y}: {v:,.0f} MW)')
print(f'\nYear-over-year change: {yoy:+.1f}%')
print(f'Real-world GB reporting: roughly -10 to -20%')
print(f'Verdict: {"✅ CONSISTENT — strong ground truth" if -25 < yoy < -8 else "⚠️ outside range — investigate before proceeding"}')

# Also YoY for the cold snap (spike event), same discipline
cold_2022 = window_mean(full, 2022, '12-08', '12-17')
cold_prior = np.nanmean([window_mean(full, y, '12-08', '12-17')
                          for y in (2018, 2019, 2021)])  # skip 2020 (COVID year)
cold_yoy = (cold_2022 - cold_prior) / cold_prior * 100
print(f'\nDec 2022 cold snap YoY (vs 2018/19/21 same dates): {cold_yoy:+.1f}%')
print('Note: 2022 also had high prices suppressing baseline demand, so a')
print('modest positive YoY still indicates genuinely elevated cold-driven load.')

# ═══════════════════════════════════════════════════════════════════════
# %% [2] Ground-truth labelling — two real events, opposite signs
# ═══════════════════════════════════════════════════════════════════════
GROUND_TRUTH_EVENTS = [
    ("2020-03-23", "2020-05-31", "COVID-19 first UK lockdown (demand collapse)"),
    ("2022-12-08", "2022-12-17", "Dec 2022 cold snap / DFS era (demand spike)"),
]

# ═══════════════════════════════════════════════════════════════════════
# %% [3] Features — Study 2 recipe, half-hourly window sizes
# ═══════════════════════════════════════════════════════════════════════
ROWS_30D = 30 * 48  # 1440 half-hourly rows = 30 days

df = full.rename(columns={'Datetime': 'Date', 'ND_MW': 'TotalLoad'}).copy()
df['return'] = df['TotalLoad'].pct_change()
df['roll_mean_30d'] = df['TotalLoad'].rolling(ROWS_30D, min_periods=48).mean()
df['roll_std_30d'] = df['TotalLoad'].rolling(ROWS_30D, min_periods=48).std()
df['roll_skew_30d'] = df['return'].rolling(ROWS_30D, min_periods=48).skew()
df = df.dropna().reset_index(drop=True)

df['is_anomaly'] = 0
for start, end, name in GROUND_TRUTH_EVENTS:
    mask = (df.Date >= start) & (df.Date <= pd.Timestamp(end) + pd.Timedelta('1D'))
    df.loc[mask, 'is_anomaly'] = 1
    print(f'Labelled {mask.sum()} rows: {name}')

df.to_csv('neso_features_labelled.csv', index=False)

# ═══════════════════════════════════════════════════════════════════════
# %% [4] Split protocols — both, matching Study 2's comparison design
# ═══════════════════════════════════════════════════════════════════════
print('\n' + '=' * 70)
print('STEP 4: Building both split protocols')
print('=' * 70)

# Protocol A: forward-chronological 70/30
cut = df.Date.quantile(0.7)
trainA = df[df.Date < cut].reset_index(drop=True)
testA = df[df.Date >= cut].reset_index(drop=True)
trainA.to_csv('neso_trainA.csv', index=False)
testA.to_csv('neso_testA.csv', index=False)
print(f'\nProtocol A (forward 70/30): split at {cut}')
print(f'  train {len(trainA)} rows, test {len(testA)} rows')

# Protocol B: clean training window containing NEITHER event.
# Window choice: 2018-06-01 to 2020-02-01 — ends before COVID, starts
# after the Feb/Mar 2018 "Beast from the East" cold event (a real GB
# demand spike that would otherwise contaminate "train on normal").
# This is a deliberate, disclosed choice; documented in methodology.
B_START, B_END = '2018-06-01', '2020-02-01'
trainB = df[(df.Date >= B_START) & (df.Date < B_END)].reset_index(drop=True)
testB = df[(df.Date < B_START) | (df.Date >= B_END)].reset_index(drop=True)
trainB.to_csv('neso_trainB.csv', index=False)
testB.to_csv('neso_testB.csv', index=False)
print(f'\nProtocol B (clean window {B_START} to {B_END}):')
print(f'  train {len(trainB)} rows, test {len(testB)} rows')

# Event reachability under each protocol
print('\nEvent reachability:')
for label, test in [('A', testA), ('B', testB)]:
    print(f'  Protocol {label}:')
    for start, end, name in GROUND_TRUTH_EVENTS:
        n = ((test.Date >= start) & (test.Date <= end)).sum()
        print(f'    {name[:45]}: {n} test rows {"✅" if n > 0 else "❌ (in training — unreachable)"}')

print('\n' + '=' * 70)
print('PHASE 2 COMPLETE — STOP. Review the YoY numbers and reachability')
print('table above. If COVID YoY is in range and both events are reachable')
print('under Protocol B, Phase 3 = the unchanged 5-method modelling script.')
print('=' * 70)
