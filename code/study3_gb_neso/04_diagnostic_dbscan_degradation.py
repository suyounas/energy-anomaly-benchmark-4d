# ═══════════════════════════════════════════════════════════════════════
# STUDY 3 — Diagnostic: WHY did DBSCAN degrade on real GB demand?
# (F1=0.068 vs 0.222/0.196 on FR load, 0.695 on WTI.)
#
# Candidate explanations to test, in order:
#   H1. eps tuning: the k-distance elbow on 29k strongly-periodic
#       training rows produces an eps that flags far too much / too
#       little of the test set.
#   H2. Periodicity dominance: daily/weekly cycles dominate the feature
#       space, so "dense normal" is structured in ways pointwise DBSCAN
#       can't separate from event rows.
#   H3. Event geometry: COVID rows are simply NOT outliers in this
#       4-feature pointwise space (level shift lives in slow rolling
#       means; half-hourly returns look normal during lockdown), so
#       sequence models see it but density methods can't.
#
# This is a DIAGNOSTIC, not a tuning exercise. The eps/min_samples sweep
# at the end measures sensitivity of the result — if some setting scores
# far better, that does NOT get promoted to a headline number unless the
# same tuning rule is re-applied uniformly across ALL FOUR datasets.
# ═══════════════════════════════════════════════════════════════════════

from google.colab import drive
drive.mount('/content/drive')

import os
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score

DATA_DIR = '/content/drive/MyDrive/neso_data'
os.chdir(DATA_DIR)
SEED = 42
np.random.seed(SEED)

FEATURES = ['return', 'roll_mean_30d', 'roll_std_30d', 'roll_skew_30d']

trainB = pd.read_csv('neso_trainB.csv', parse_dates=['Date'])
testB = pd.read_csv('neso_testB.csv', parse_dates=['Date'])
print(f'trainB {len(trainB)} rows, testB {len(testB)} rows')

scaler = StandardScaler().fit(trainB[FEATURES])
X_train = scaler.transform(trainB[FEATURES])
X_test = scaler.transform(testB[FEATURES])
y_test = testB['is_anomaly'].values

# ═══════════════════════════════════════════════════════════════════════
# %% [H1] What eps did the tuning rule pick, and what flag rate results?
# ═══════════════════════════════════════════════════════════════════════
print('\n' + '=' * 70)
print('H1: eps tuning and flag rate')
print('=' * 70)

knn = NearestNeighbors(n_neighbors=5).fit(X_train)
distances, _ = knn.kneighbors(X_train)
kdist = np.sort(distances[:, -1])
eps_used = np.percentile(kdist, 90)
print(f'k-distance (k=5) percentiles on trainB:')
for p in [50, 75, 90, 95, 99]:
    print(f'  p{p}: {np.percentile(kdist, p):.4f}')
print(f'eps used in Phase 3 (p90 rule): {eps_used:.4f}')
print(f'(Study 1 reference: WTI eps=0.441, Henry Hub eps=0.596 — daily data)')

db = DBSCAN(eps=eps_used, min_samples=5).fit(X_test)
flag = (db.labels_ == -1).astype(int)
flag_rate = flag.mean()
print(f'\nDBSCAN flag rate on testB: {flag_rate:.1%} of {len(X_test)} rows')
print(f'Anomaly base rate in testB: {y_test.mean():.1%}')
print(f'Number of clusters found (excl. noise): {len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)}')
p, r, f1 = (precision_score(y_test, flag, zero_division=0),
            recall_score(y_test, flag, zero_division=0),
            f1_score(y_test, flag, zero_division=0))
print(f'Reproduced Phase 3 result: P={p:.3f} R={r:.3f} F1={f1:.3f}')

# ═══════════════════════════════════════════════════════════════════════
# %% [H2] Periodicity dominance: how much feature variance is explained
# by hour-of-day and day-of-week alone?
# ═══════════════════════════════════════════════════════════════════════
print('\n' + '=' * 70)
print('H2: variance explained by daily/weekly cycles (R^2 of group means)')
print('=' * 70)

tb = testB.copy()
tb['hour'] = tb.Date.dt.hour * 2 + (tb.Date.dt.minute // 30)  # 0-47 slot
tb['dow'] = tb.Date.dt.dayofweek

for feat in FEATURES:
    total_var = tb[feat].var()
    hour_means = tb.groupby('hour')[feat].transform('mean')
    dow_hour_means = tb.groupby(['dow', 'hour'])[feat].transform('mean')
    r2_hour = hour_means.var() / total_var
    r2_dowhour = dow_hour_means.var() / total_var
    print(f'  {feat:<15} R2(slot-of-day)={r2_hour:.3f}   R2(dow x slot)={r2_dowhour:.3f}')
print('\nInterpretation: values near 1 mean the feature is dominated by')
print('the deterministic daily/weekly cycle; near 0 means cycle-free.')
print('Compare: FR load in Study 2 was hourly with the same features —')
print('if the GB half-hourly "return" is cycle-dominated, pointwise')
print('density in this space is structured by time-of-day, not by')
print('normal-vs-anomalous state.')

# ═══════════════════════════════════════════════════════════════════════
# %% [H3] Event geometry: are COVID rows even outliers in this space?
# ═══════════════════════════════════════════════════════════════════════
print('\n' + '=' * 70)
print('H3: where do event rows sit in the standardized feature space?')
print('=' * 70)

covid_mask = (testB.Date >= '2020-03-23') & (testB.Date <= '2020-05-31')
cold_mask = (testB.Date >= '2022-12-08') & (testB.Date <= '2022-12-17')
normal_mask = (testB['is_anomaly'] == 0)

def profile(mask, label):
    Xm = X_test[mask.values]
    print(f'\n  {label} ({mask.sum()} rows) — mean |z| per feature vs train:')
    for i, feat in enumerate(FEATURES):
        print(f'    {feat:<15} mean z = {Xm[:, i].mean():+.2f}   mean |z| = {np.abs(Xm[:, i]).mean():.2f}')
    # distance to 5th nearest TRAIN neighbour (the density DBSCAN sees)
    d, _ = knn.kneighbors(Xm)
    print(f'    5-NN distance to train: median={np.median(d[:, -1]):.3f} '
          f'(train median: {np.median(kdist):.3f}, eps={eps_used:.3f})')
    print(f'    fraction of rows with 5-NN distance > eps: {(d[:, -1] > eps_used).mean():.1%}')

profile(pd.Series(normal_mask.values), 'NORMAL test rows')
profile(covid_mask, 'COVID rows')
profile(cold_mask, 'COLD SNAP rows')

print('\nInterpretation: if COVID rows have 5-NN distances similar to')
print('normal rows (fraction > eps close to normal rows\' fraction), they')
print('are simply not density outliers pointwise — the level shift lives')
print('in slowly-drifting rolling means while half-hourly returns stay')
print('normal, which is exactly what sequence models can see and')
print('pointwise density methods cannot. That would make the DBSCAN')
print('degradation a genuine structural finding about method-data fit.')

# ═══════════════════════════════════════════════════════════════════════
# %% [Sensitivity] eps / min_samples sweep — diagnostic only
# ═══════════════════════════════════════════════════════════════════════
print('\n' + '=' * 70)
print('SENSITIVITY SWEEP (diagnostic only — see header note)')
print('=' * 70)
print(f'{"eps rule":<12}{"eps":>8}{"min_s":>7}{"flag%":>8}{"P":>8}{"R":>8}{"F1":>8}')
for pct in [75, 80, 85, 90, 95]:
    eps = np.percentile(kdist, pct)
    for ms in [5, 20, 50]:
        db_s = DBSCAN(eps=eps, min_samples=ms).fit(X_test)
        fl = (db_s.labels_ == -1).astype(int)
        print(f'{"p"+str(pct):<12}{eps:>8.3f}{ms:>7}{fl.mean():>8.1%}'
              f'{precision_score(y_test, fl, zero_division=0):>8.3f}'
              f'{recall_score(y_test, fl, zero_division=0):>8.3f}'
              f'{f1_score(y_test, fl, zero_division=0):>8.3f}')

print('\nDiagnostic complete. Send the full output back for interpretation.')
print('Reminder: nothing from the sweep becomes a headline number unless')
print('the same rule is re-applied uniformly to all four datasets.')
