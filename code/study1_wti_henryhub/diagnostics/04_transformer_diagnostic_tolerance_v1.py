# Run in the same Colab session. Retrains all 4 neural models fresh
# (rather than trusting leftover local variables) to avoid the state bugs
# we've already hit twice. Takes ~6-10 min total on CPU.
#
# Requires these to still exist in the session from the ORIGINAL script:
# SeqDataset, LSTMAutoencoder, TransformerAutoencoder, train_lstm_ae,
# train_transformer, lstm_ae_scores, transformer_scores, FEATURES,
# GROUND_TRUTH_EVENTS, SEQ_LEN, device, wti_train/test, hh_train/test.
# If any NameError appears below, the original script's cell needs
# re-running first.

import torch.nn as nn  # re-assert in case of any earlier collision
from sklearn.neighbors import NearestNeighbors

required = ["SeqDataset", "train_lstm_ae", "train_transformer",
            "lstm_ae_scores", "transformer_scores", "FEATURES",
            "GROUND_TRUTH_EVENTS", "SEQ_LEN", "wti_train", "wti_test",
            "hh_train", "hh_test"]
missing = [r for r in required if r not in dir()]
if missing:
    raise RuntimeError(f"Missing from session: {missing}. Re-run the original "
                        f"colab_real_training.py cell first, then this script.")

print("=" * 60)
print("STEP 1: Retrain all four neural models fresh, save globally")
print("=" * 60)

scaler_wti3 = StandardScaler().fit(wti_train[FEATURES])
scaler_hh3  = StandardScaler().fit(hh_train[FEATURES])

wti_train_ds3 = SeqDataset(wti_train, FEATURES, scaler_wti3)
wti_test_ds3  = SeqDataset(wti_test, FEATURES, scaler_wti3)
hh_train_ds3  = SeqDataset(hh_train, FEATURES, scaler_hh3)
hh_test_ds3   = SeqDataset(hh_test, FEATURES, scaler_hh3)

print("Training WTI LSTM...")
wti_lstm_model = train_lstm_ae(wti_train_ds3, n_features=len(FEATURES))
print("Training WTI Transformer...")
wti_trans_model = train_transformer(wti_train_ds3, n_features=len(FEATURES))
print("Training Henry Hub LSTM...")
hh_lstm_model = train_lstm_ae(hh_train_ds3, n_features=len(FEATURES))
print("Training Henry Hub Transformer...")
hh_trans_model = train_transformer(hh_train_ds3, n_features=len(FEATURES))

wti_lstm_scores  = lstm_ae_scores(wti_lstm_model, wti_test_ds3)
wti_trans_scores = transformer_scores(wti_trans_model, wti_test_ds3)
hh_lstm_scores   = lstm_ae_scores(hh_lstm_model, hh_test_ds3)
hh_trans_scores  = transformer_scores(hh_trans_model, hh_test_ds3)

print("\n" + "=" * 60)
print("STEP 2: Transformer percentile-rank diagnostic on Henry Hub "
      "(symmetric to the LSTM check)")
print("=" * 60)

y_true_seq_hh = hh_test["is_anomaly"].values[SEQ_LEN:]
true_pos_hh = np.where(y_true_seq_hh == 1)[0]

thresh_trans_hh = np.percentile(hh_trans_scores, 95)
flagged_trans_hh = np.where(hh_trans_scores > thresh_trans_hh)[0]
dates_seq_hh = pd.to_datetime(hh_test["Date"].values[SEQ_LEN:])

print(f"Transformer reconstruction error: min={hh_trans_scores.min():.4f}, "
      f"max={hh_trans_scores.max():.4f}, 95th pct threshold={thresh_trans_hh:.4f}")
print(f"Days flagged: {len(flagged_trans_hh)}")
if len(flagged_trans_hh) > 0:
    fd = dates_seq_hh[flagged_trans_hh]
    print(f"Flagged date range: {fd.min().date()} to {fd.max().date()}")

print(f"True positives at strict threshold: {y_true_seq_hh[flagged_trans_hh].sum()} "
      f"out of {y_true_seq_hh.sum()}")

if len(true_pos_hh) > 0:
    scores_on_true = hh_trans_scores[true_pos_hh]
    pct_ranks = [(hh_trans_scores < s).mean() * 100 for s in scores_on_true]
    print(f"\nTransformer reconstruction error PERCENTILE RANK of true anomaly days:")
    print(f"  mean: {np.mean(pct_ranks):.1f}  min: {np.min(pct_ranks):.1f}  "
          f"max: {np.max(pct_ranks):.1f}")
    print(f"  (compare to LSTM's earlier result: mean 61.5, max 93.2 — "
          f"{'similar pattern' if abs(np.mean(pct_ranks) - 61.5) < 15 else 'notably different pattern'})")

print("\n" + "=" * 60)
print("STEP 3: Classical methods (Isolation Forest, DBSCAN w/ fixed eps, "
      "rolling z-score)")
print("=" * 60)

def get_classical_flags(train_df, test_df, feature_cols):
    scaler = StandardScaler().fit(train_df[feature_cols])
    X_train = scaler.transform(train_df[feature_cols])
    X_test = scaler.transform(test_df[feature_cols])

    iso = IsolationForest(contamination=0.05, random_state=SEED).fit(X_train)
    iso_scores = -iso.score_samples(X_test)
    iso_flag = (iso_scores > np.percentile(iso_scores, 95)).astype(int)

    knn_model = NearestNeighbors(n_neighbors=5).fit(X_train)
    distances, _ = knn_model.kneighbors(X_train)
    proper_eps = np.percentile(np.sort(distances[:, -1]), 90)
    db = DBSCAN(eps=proper_eps, min_samples=5).fit(X_test)
    db_flag = (db.labels_ == -1).astype(int)

    z_flag = (test_df["zscore"].abs().fillna(0).values > 3).astype(int)

    return iso_flag, db_flag, z_flag

wti_iso_flag, wti_db_flag, wti_z_flag = get_classical_flags(wti_train, wti_test, FEATURES)
hh_iso_flag,  hh_db_flag,  hh_z_flag  = get_classical_flags(hh_train, hh_test, FEATURES)

print("\n" + "=" * 60)
print("STEP 4: Tolerance-window metric (+/- 10 days), same protocol for "
      "every method, alongside the strict same-day metric")
print("=" * 60)

TOLERANCE_DAYS = 10

def build_tolerance_labels(dates, events, tolerance_days=TOLERANCE_DAYS):
    dates = pd.to_datetime(pd.Series(dates))
    labels = np.zeros(len(dates), dtype=int)
    for start, end in events:
        start_ext = pd.Timestamp(start) - pd.Timedelta(days=tolerance_days)
        end_ext = pd.Timestamp(end) + pd.Timedelta(days=tolerance_days)
        mask = (dates >= start_ext) & (dates <= end_ext)
        labels[mask.values] = 1
    return labels

def eval_both(y_true_strict, y_true_tol, y_pred):
    p_s = precision_score(y_true_strict, y_pred, zero_division=0)
    r_s = recall_score(y_true_strict, y_pred, zero_division=0)
    f1_s = f1_score(y_true_strict, y_pred, zero_division=0)
    p_t = precision_score(y_true_tol, y_pred, zero_division=0)
    r_t = recall_score(y_true_tol, y_pred, zero_division=0)
    f1_t = f1_score(y_true_tol, y_pred, zero_division=0)
    return p_s, r_s, f1_s, p_t, r_t, f1_t

# Full-length (day-level) labels for classical methods
wti_true_strict_full = wti_test["is_anomaly"].values
hh_true_strict_full  = hh_test["is_anomaly"].values
wti_true_tol_full = build_tolerance_labels(wti_test["Date"].values, GROUND_TRUTH_EVENTS)
hh_true_tol_full  = build_tolerance_labels(hh_test["Date"].values, GROUND_TRUTH_EVENTS)

# Sequence-offset labels for neural methods
wti_true_strict_seq = wti_test["is_anomaly"].values[SEQ_LEN:]
hh_true_strict_seq  = hh_test["is_anomaly"].values[SEQ_LEN:]
wti_true_tol_seq = build_tolerance_labels(wti_test["Date"].values[SEQ_LEN:], GROUND_TRUTH_EVENTS)
hh_true_tol_seq  = build_tolerance_labels(hh_test["Date"].values[SEQ_LEN:], GROUND_TRUTH_EVENTS)

def flag_from_scores(scores, pct=95):
    return (scores > np.percentile(scores, pct)).astype(int)

wti_lstm_flag  = flag_from_scores(wti_lstm_scores)
wti_trans_flag = flag_from_scores(wti_trans_scores)
hh_lstm_flag   = flag_from_scores(hh_lstm_scores)
hh_trans_flag  = flag_from_scores(hh_trans_scores)

rows = []
combos = [
    ("WTI", "LSTM Autoencoder", wti_true_strict_seq, wti_true_tol_seq, wti_lstm_flag),
    ("WTI", "Transformer Encoder", wti_true_strict_seq, wti_true_tol_seq, wti_trans_flag),
    ("WTI", "Isolation Forest", wti_true_strict_full, wti_true_tol_full, wti_iso_flag),
    ("WTI", "DBSCAN", wti_true_strict_full, wti_true_tol_full, wti_db_flag),
    ("WTI", "Rolling Z-score", wti_true_strict_full, wti_true_tol_full, wti_z_flag),
    ("Henry Hub", "LSTM Autoencoder", hh_true_strict_seq, hh_true_tol_seq, hh_lstm_flag),
    ("Henry Hub", "Transformer Encoder", hh_true_strict_seq, hh_true_tol_seq, hh_trans_flag),
    ("Henry Hub", "Isolation Forest", hh_true_strict_full, hh_true_tol_full, hh_iso_flag),
    ("Henry Hub", "DBSCAN", hh_true_strict_full, hh_true_tol_full, hh_db_flag),
    ("Henry Hub", "Rolling Z-score", hh_true_strict_full, hh_true_tol_full, hh_z_flag),
]

for commodity, method, y_s, y_t, y_pred in combos:
    p_s, r_s, f1_s, p_t, r_t, f1_t = eval_both(y_s, y_t, y_pred)
    rows.append({
        "Commodity": commodity, "Method": method,
        "Strict_Precision": round(p_s, 3), "Strict_Recall": round(r_s, 3), "Strict_F1": round(f1_s, 3),
        f"Tol{TOLERANCE_DAYS}d_Precision": round(p_t, 3), f"Tol{TOLERANCE_DAYS}d_Recall": round(r_t, 3),
        f"Tol{TOLERANCE_DAYS}d_F1": round(f1_t, 3),
    })

final_df = pd.DataFrame(rows)
print(final_df.to_string(index=False))
final_df.to_csv("real_model_comparison_with_tolerance.csv", index=False)
print("\nSaved: real_model_comparison_with_tolerance.csv")

from google.colab import files
files.download("real_model_comparison_with_tolerance.csv")
