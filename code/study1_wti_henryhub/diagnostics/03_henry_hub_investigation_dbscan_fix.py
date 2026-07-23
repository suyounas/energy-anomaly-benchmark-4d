# Run in the same Colab session — self-contained, doesn't rely on fragile
# leftover local variables from run_pipeline(). Reuses only the genuinely
# global objects: wti_train/test, hh_train/test, FEATURES, GROUND_TRUTH_EVENTS,
# SEQ_LEN, device, and the class/function definitions.

from sklearn.neighbors import NearestNeighbors  # <-- was missing, caused NameError in FIX 2

print("=" * 60)
print("FIX 1: Retrain Henry Hub LSTM explicitly, inspect flagged dates")
print("=" * 60)

scaler_hh2 = StandardScaler().fit(hh_train[FEATURES])
hh_train_ds2 = SeqDataset(hh_train, FEATURES, scaler_hh2)
hh_test_ds2  = SeqDataset(hh_test, FEATURES, scaler_hh2)

print("Retraining Henry Hub LSTM Autoencoder...")
hh_lstm_model = train_lstm_ae(hh_train_ds2, n_features=len(FEATURES))
hh_scores2 = lstm_ae_scores(hh_lstm_model, hh_test_ds2)

dates_seq = hh_test["Date"].values[SEQ_LEN:]
y_true_seq = hh_test["is_anomaly"].values[SEQ_LEN:]

thresh = np.percentile(hh_scores2, 95)
flagged_idx = np.where(hh_scores2 > thresh)[0]
flagged_dates = pd.to_datetime(dates_seq[flagged_idx])

print(f"\nReconstruction error stats: min={hh_scores2.min():.4f}, "
      f"max={hh_scores2.max():.4f}, 95th pct threshold={thresh:.4f}")
print(f"Number of days flagged: {len(flagged_dates)}")
if len(flagged_dates) > 0:
    print(f"Flagged date range: {flagged_dates.min().date()} to {flagged_dates.max().date()}")
    print(f"\nAll flagged dates:")
    for d in sorted(flagged_dates):
        print(f"  {d.date()}")
else:
    print("WARNING: zero dates flagged at the 95th percentile threshold — check hh_scores2 for NaNs/constant values.")

print(f"\nGround truth event windows (for comparison):")
for start, end in GROUND_TRUTH_EVENTS:
    print(f"  {start} to {end}")

print(f"\nTrue positives at 95th pct threshold: {y_true_seq[flagged_idx].sum()} "
      f"out of {y_true_seq.sum()} actual positives in test set")

# Also check: what error score did the model actually assign to the TRUE
# anomaly days, regardless of whether they cleared the threshold?
true_anomaly_positions = np.where(y_true_seq == 1)[0]
if len(true_anomaly_positions) > 0:
    scores_on_true_anomalies = hh_scores2[true_anomaly_positions]
    percentile_of_true_anomalies = [
        (hh_scores2 < s).mean() * 100 for s in scores_on_true_anomalies
    ]
    print(f"\nReconstruction error PERCENTILE RANK of the actual anomaly days:")
    print(f"  (if the model were working, these should be near 95-100)")
    print(f"  mean percentile rank: {np.mean(percentile_of_true_anomalies):.1f}")
    print(f"  min: {np.min(percentile_of_true_anomalies):.1f}, "
          f"max: {np.max(percentile_of_true_anomalies):.1f}")
else:
    print("\nNo true anomaly positions found in y_true_seq — check label alignment.")

print("\n" + "=" * 60)
print("FIX 2: Rerun DBSCAN with properly-tuned eps (from k-distance elbow)")
print("=" * 60)

for name, tr, te, y_col in [("WTI", wti_train, wti_test, "is_anomaly"),
                             ("Henry Hub", hh_train, hh_test, "is_anomaly")]:
    scaler_fix = StandardScaler().fit(tr[FEATURES])
    X_tr_fix = scaler_fix.transform(tr[FEATURES])
    X_te_fix = scaler_fix.transform(te[FEATURES])

    nn_fix = NearestNeighbors(n_neighbors=5).fit(X_tr_fix)
    distances_fix, _ = nn_fix.kneighbors(X_tr_fix)
    proper_eps = np.percentile(np.sort(distances_fix[:, -1]), 90)

    db_fix = DBSCAN(eps=proper_eps, min_samples=5).fit(X_te_fix)
    y_pred_fix = (db_fix.labels_ == -1).astype(int)
    y_true_fix = te[y_col].values

    p = precision_score(y_true_fix, y_pred_fix, zero_division=0)
    r = recall_score(y_true_fix, y_pred_fix, zero_division=0)
    f1 = f1_score(y_true_fix, y_pred_fix, zero_division=0)

    n_noise = (db_fix.labels_ == -1).sum()
    print(f"\n{name}: eps={proper_eps:.3f}  ->  "
          f"{n_noise}/{len(te)} points flagged as noise/anomaly")
    print(f"  Precision={p:.3f}  Recall={r:.3f}  F1={f1:.3f}  "
          f"(vs. F1=0.000 with the original guessed eps=1.5)")
