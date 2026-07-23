# ═══════════════════════════════════════════════════════════════════════
# Phase 2b (SELF-CONTAINED): Ground truth, features, train/test split
# Loads fresh from the saved national_load_*_full.csv files on Drive —
# does NOT depend on Phase 2a's in-memory variables still being around.
# ═══════════════════════════════════════════════════════════════════════

# %% [0] Mount Drive (safe no-op if already mounted) ------------------------
from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import numpy as np
import os

DATA_DIR = '/content/drive/MyDrive/grid_data'
os.chdir(DATA_DIR)
print(f'Working directory: {os.getcwd()}')

# %% [1] Load country series fresh from saved CSVs ---------------------------
TARGET_COUNTRIES = ['FR', 'DE', 'CH', 'IT', 'ES']

country_dfs = {}
for country in TARGET_COUNTRIES:
    fname = f'national_load_{country}_full.csv'
    if not os.path.exists(fname):
        print(f'⚠️ {fname} not found — check Phase 2a completed for this country.')
        continue
    df = pd.read_csv(fname, parse_dates=['Date'])
    country_dfs[country] = df
    print(f'{country}: loaded {len(df)} rows, {df.Date.min()} to {df.Date.max()}')

# %% [2] Known data gap — document it, don't silently ignore it -------------
# Confirmed in Phase 2a's boundary check: a 7-day-1-hour gap exists between
# 2018-12-30 23:00 and 2019-01-07 00:00 (structural artifact of concatenating
# 364-day year files against real calendar years). Neither ground-truth
# event falls near this gap, but rolling-window features computed across it
# will splice non-adjacent real time together for ~30 days surrounding it.
KNOWN_DATA_GAP = (pd.Timestamp('2018-12-30 23:00:00'), pd.Timestamp('2019-01-07 00:00:00'))
print(f'\nKnown data gap: {KNOWN_DATA_GAP[0]} to {KNOWN_DATA_GAP[1]} '
      f'({(KNOWN_DATA_GAP[1]-KNOWN_DATA_GAP[0])})')
print('Neither ground-truth event falls near this gap — documented, not fixed.')

# %% [3] Ground truth events -------------------------------------------------
GROUND_TRUTH_EVENTS = [
    ("2016-10-10", "2017-01-27", "French nuclear crisis"),
    ("2020-03-16", "2020-05-11", "COVID-19 demand collapse"),
]
print('\nGround truth events:')
for start, end, name in GROUND_TRUTH_EVENTS:
    print(f'  {name}: {start} to {end}')

# %% [4] Feature engineering (same recipe as WTI/Henry Hub, hourly-adjusted)
# 30-day rolling window on hourly data = 720 hours
FEATURES = ['return', 'roll_mean_30d', 'roll_std_30d', 'roll_skew_30d']

def build_features(df, value_col='TotalLoad'):
    df = df.copy().sort_values('Date').reset_index(drop=True)
    df['return'] = df[value_col].pct_change()
    df['roll_mean_30d'] = df[value_col].rolling(720, min_periods=24).mean()
    df['roll_std_30d'] = df[value_col].rolling(720, min_periods=24).std()
    df['roll_skew_30d'] = df['return'].rolling(720, min_periods=24).skew()
    df = df.dropna().reset_index(drop=True)
    return df

def label_ground_truth(df):
    df = df.copy()
    df['is_anomaly'] = 0
    for start, end, name in GROUND_TRUTH_EVENTS:
        mask = (df.Date >= start) & (df.Date <= end)
        df.loc[mask, 'is_anomaly'] = 1
    return df

featured_dfs = {}
for country in TARGET_COUNTRIES:
    if country not in country_dfs:
        continue
    df_feat = build_features(country_dfs[country])
    df_feat = label_ground_truth(df_feat)
    featured_dfs[country] = df_feat
    n_pos = df_feat['is_anomaly'].sum()
    print(f'{country}: {len(df_feat)} rows after feature engineering, '
          f'{n_pos} labelled anomaly-window rows')

# %% [5] Chronological train/test split (70/30) ------------------------------
def chrono_split(df, train_frac=0.7):
    n = len(df)
    cut = int(n * train_frac)
    return df.iloc[:cut].reset_index(drop=True), df.iloc[cut:].reset_index(drop=True)

train_test = {}
for country in TARGET_COUNTRIES:
    if country not in featured_dfs:
        continue
    train, test = chrono_split(featured_dfs[country])
    train_test[country] = (train, test)
    print(f'{country}: train {train.Date.min()} to {train.Date.max()} ({len(train)} rows), '
          f'test {test.Date.min()} to {test.Date.max()} ({len(test)} rows)')

# %% [6] CRITICAL — event reachability check (same lesson as WTI/Henry Hub) -
print('\n' + '=' * 70)
print('EVENT REACHABILITY CHECK — which events fall in the TEST set')
print('=' * 70)
for country in TARGET_COUNTRIES:
    if country not in train_test:
        continue
    _, test = train_test[country]
    print(f'\n{country}:')
    for start, end, name in GROUND_TRUTH_EVENTS:
        in_test = ((test.Date >= start) & (test.Date <= end)).sum()
        in_train = ((train_test[country][0].Date >= start) & (train_test[country][0].Date <= end)).sum()
        status = '✅ reachable' if in_test > 0 else '❌ NOT reachable (entirely in train)'
        print(f'  {name}: {in_test} rows in test, {in_train} rows in train  {status}')

# %% [7] Save train/test splits to Drive (survives disconnects) -------------
for country in TARGET_COUNTRIES:
    if country not in train_test:
        continue
    train, test = train_test[country]
    train.to_csv(f'train_{country}.csv', index=False)
    test.to_csv(f'test_{country}.csv', index=False)

print(f'\nSaved train/test splits for {len(train_test)} countries to {DATA_DIR}')
print('\nPhase 2b complete. Review the event reachability check (step 6) —')
print('any event marked NOT reachable for a country should be excluded from')
print('that country\'s evaluation, same as we found with WTI/Henry Hub.')
