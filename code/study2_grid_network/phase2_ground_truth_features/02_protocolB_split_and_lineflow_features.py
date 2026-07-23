# ═══════════════════════════════════════════════════════════════════════
# Phase 3a: Second split protocol (clean training window) + line-flow
# feature engineering, so we can compare BOTH protocols on load data and
# properly prepare the nuclear-crisis line-flow data for modelling.
# ═══════════════════════════════════════════════════════════════════════

from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import numpy as np
import os

DATA_DIR = '/content/drive/MyDrive/grid_data'
os.chdir(DATA_DIR)

GROUND_TRUTH_EVENTS = [
    ("2016-10-10", "2017-01-27", "French nuclear crisis"),
    ("2020-03-16", "2020-05-11", "COVID-19 demand collapse"),
]

# %% [1] PROTOCOL B: clean-training-window split for LOAD data --------------
# Train on a window containing NEITHER event; evaluate on everything else
# (both before and after). This is the protocol that makes both events
# reachable, unlike the forward-only 70/30 split.

FEATURES = ['return', 'roll_mean_30d', 'roll_std_30d', 'roll_skew_30d']
TRAIN_WINDOW = ('2017-06-01', '2019-06-01')  # clear of both events, real margin

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

def clean_window_split(df, train_start, train_end):
    train = df[(df.Date >= train_start) & (df.Date < train_end)].reset_index(drop=True)
    test = df[(df.Date < train_start) | (df.Date >= train_end)].reset_index(drop=True)
    return train, test

TARGET_COUNTRIES = ['FR', 'DE', 'CH', 'IT', 'ES']
protocolB_splits = {}

for country in TARGET_COUNTRIES:
    fname = f'national_load_{country}_full.csv'
    df_raw = pd.read_csv(fname, parse_dates=['Date'])
    df_feat = build_features(df_raw)
    df_feat = label_ground_truth(df_feat)
    train, test = clean_window_split(df_feat, *TRAIN_WINDOW)
    protocolB_splits[country] = (train, test)
    train.to_csv(f'trainB_{country}.csv', index=False)
    test.to_csv(f'testB_{country}.csv', index=False)
    print(f'{country}: Protocol B — train {len(train)} rows ({TRAIN_WINDOW[0]} to '
          f'{TRAIN_WINDOW[1]}), test {len(test)} rows (everything else)')

# %% [2] Event reachability check under Protocol B --------------------------
print('\n' + '=' * 70)
print('PROTOCOL B EVENT REACHABILITY (should show BOTH events now)')
print('=' * 70)
for country in TARGET_COUNTRIES:
    _, test = protocolB_splits[country]
    print(f'\n{country}:')
    for start, end, name in GROUND_TRUTH_EVENTS:
        in_test = ((test.Date >= start) & (test.Date <= end)).sum()
        status = '✅' if in_test > 0 else '❌'
        print(f'  {name}: {in_test} rows in test  {status}')

# %% [3] Feature engineering for LINE-FLOW data (nuclear crisis) ------------
# Same recipe, applied to france_total_net_export.csv. Only the nuclear
# crisis event is relevant here (COVID is a load story, not a flow story,
# per our earlier finding) — but we keep both labels for completeness,
# the COVID one will just show near-zero anomaly signal here as expected.

line_df_raw = pd.read_csv('france_total_net_export.csv', parse_dates=['Date'])
line_df_feat = build_features(line_df_raw, value_col='TotalNetExport')
line_df_feat = label_ground_truth(line_df_feat)

# Clean training window for line data — same window, confirmed clear of
# both events, and clear of the small amount of 2018 data we have too
train_line, test_line = clean_window_split(line_df_feat, '2017-06-01', '2018-01-01')
# Note: shorter window since we only have data through 2018 for lines —
# this is intentionally asymmetric vs. the load data's window

train_line.to_csv('trainB_lineflow.csv', index=False)
test_line.to_csv('testB_lineflow.csv', index=False)

print('\n' + '=' * 70)
print('LINE-FLOW (nuclear crisis) — Protocol B split')
print('=' * 70)
print(f'Train: {len(train_line)} rows, {train_line.Date.min()} to {train_line.Date.max()}')
print(f'Test: {len(test_line)} rows, spanning before+after train window')
for start, end, name in GROUND_TRUTH_EVENTS:
    in_test = ((test_line.Date >= start) & (test_line.Date <= end)).sum()
    print(f'  {name} reachable in test: {in_test} rows')

print('\nPhase 3a complete. Both protocols now available for load data;')
print('line-flow data prepared with Protocol B (the only viable one, given')
print('the data only spans 2016-2018 and both boundary years matter).')
