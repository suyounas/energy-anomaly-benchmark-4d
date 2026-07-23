# Run this in the SAME Colab session, in a new cell, right after the main
# script has finished — it reuses variables already in memory
# (wti_train, wti_test, hh_train, hh_test, GROUND_TRUTH_EVENTS, etc.)

print("=" * 60)
print("DIAGNOSTIC 1: Date ranges per split")
print("=" * 60)
for name, tr, te in [("WTI", wti_train, wti_test), ("Henry Hub", hh_train, hh_test)]:
    print(f"\n{name}")
    print(f"  Train: {tr.Date.min().date()} to {tr.Date.max().date()}  ({len(tr)} rows)")
    print(f"  Test:  {te.Date.min().date()} to {te.Date.max().date()}  ({len(te)} rows)")

print("\n" + "=" * 60)
print("DIAGNOSTIC 2: Which ground-truth events actually fall in TEST vs TRAIN")
print("=" * 60)
for name, tr, te in [("WTI", wti_train, wti_test), ("Henry Hub", hh_train, hh_test)]:
    print(f"\n{name}")
    for start, end in GROUND_TRUTH_EVENTS:
        start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
        in_test  = ((te.Date >= start_ts) & (te.Date <= end_ts)).sum()
        in_train = ((tr.Date >= start_ts) & (tr.Date <= end_ts)).sum()
        print(f"  {start} to {end}:  {in_train} rows in TRAIN, {in_test} rows in TEST")

print("\n" + "=" * 60)
print("DIAGNOSTIC 3: Total positive labels available in each test set")
print("=" * 60)
print(f"WTI test positives:       {wti_test['is_anomaly'].sum()} / {len(wti_test)}")
print(f"Henry Hub test positives: {hh_test['is_anomaly'].sum()} / {len(hh_test)}")

print("\n" + "=" * 60)
print("DIAGNOSTIC 4: DBSCAN cluster label distribution on WTI test set")
print("=" * 60)
scaler_check = StandardScaler().fit(wti_train[FEATURES])
X_test_check = scaler_check.transform(wti_test[FEATURES])
from sklearn.cluster import DBSCAN as DBSCAN_check
db_check = DBSCAN_check(eps=1.5, min_samples=5).fit(X_test_check)
unique, counts = np.unique(db_check.labels_, return_counts=True)
for u, c in zip(unique, counts):
    label_name = "noise/anomaly (-1)" if u == -1 else f"cluster {u}"
    print(f"  {label_name}: {c} points")
