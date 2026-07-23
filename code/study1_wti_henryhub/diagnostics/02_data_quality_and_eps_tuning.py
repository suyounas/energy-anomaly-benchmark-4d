# Run in the same Colab session — investigates the Henry Hub neural-net zero
# result and finds a properly-tuned eps for DBSCAN instead of the guessed 1.5

print("=" * 60)
print("DIAGNOSTIC 5: Basic data quality check — Henry Hub vs WTI")
print("=" * 60)
for name, df in [("WTI", wti_f), ("Henry Hub", hh_f)]:
    print(f"\n{name} feature summary (full series):")
    print(df[FEATURES].describe().loc[["mean", "std", "min", "max"]])
    n_zero_std = (df["roll_std_30"] == 0).sum()
    print(f"  Rows with roll_std_30 == 0 (flat price, degenerate z-score): {n_zero_std}")

print("\n" + "=" * 60)
print("DIAGNOSTIC 6: Where do the highest-error days actually fall for Henry Hub?")
print("=" * 60)
# Recompute Henry Hub test-set reconstruction error and inspect the top 5%
scaler_hh_check = StandardScaler().fit(hh_train[FEATURES])
hh_test_ds_check = SeqDataset(hh_test, FEATURES, scaler_hh_check)
hh_scores_check = lstm_ae_scores(lstm_model, hh_test_ds_check) if 'lstm_model' in dir() else None

if hh_scores_check is not None:
    dates_seq = hh_test["Date"].values[SEQ_LEN:]
    thresh = np.percentile(hh_scores_check, 95)
    flagged_idx = np.where(hh_scores_check > thresh)[0]
    flagged_dates = pd.to_datetime(dates_seq[flagged_idx])
    print(f"Threshold (95th pct of reconstruction error): {thresh:.4f}")
    print(f"Number of days flagged: {len(flagged_dates)}")
    print(f"Flagged date range: {flagged_dates.min()} to {flagged_dates.max()}")
    print(f"\nFirst 10 flagged dates:")
    print(sorted(flagged_dates)[:10])
    print(f"\nGround truth event windows (for comparison):")
    for start, end in GROUND_TRUTH_EVENTS:
        print(f"  {start} to {end}")
else:
    print("Re-run this after the main script — lstm_model must exist in memory.")
    print("If it's gone (session restarted), retrain: lstm_model = train_lstm_ae(SeqDataset(hh_train, FEATURES, scaler_hh_check), n_features=len(FEATURES))")

print("\n" + "=" * 60)
print("DIAGNOSTIC 7: Properly tuned eps for DBSCAN (k-distance elbow, train set only)")
print("=" * 60)
from sklearn.neighbors import NearestNeighbors

for name, tr in [("WTI", wti_train), ("Henry Hub", hh_train)]:
    scaler_tmp = StandardScaler().fit(tr[FEATURES])
    X_tr = scaler_tmp.transform(tr[FEATURES])
    k = 5  # matches min_samples
    knn_model = NearestNeighbors(n_neighbors=k).fit(X_tr)
    distances, _ = knn_model.kneighbors(X_tr)
    k_dist = np.sort(distances[:, -1])
    # a reasonable elbow heuristic: look at the 90th percentile of k-distances
    suggested_eps = np.percentile(k_dist, 90)
    print(f"{name}: suggested eps (90th pct of {k}-NN distance) = {suggested_eps:.3f}  "
          f"(vs. the guessed eps=1.5 used originally)")
