# Run in the same Colab session. Reuses the flag arrays already computed
# in the previous script (wti_lstm_flag, wti_trans_flag, wti_iso_flag,
# wti_db_flag, wti_z_flag, and the Henry Hub equivalents) — no retraining
# needed, this just recomputes the tolerance metric correctly.

required = ["wti_lstm_flag", "wti_trans_flag", "wti_iso_flag", "wti_db_flag", "wti_z_flag",
            "hh_lstm_flag", "hh_trans_flag", "hh_iso_flag", "hh_db_flag", "hh_z_flag",
            "GROUND_TRUTH_EVENTS", "SEQ_LEN", "wti_test", "hh_test"]
missing = [r for r in required if r not in dir()]
if missing:
    raise RuntimeError(f"Missing from session: {missing}. Re-run the previous "
                        f"tolerance_and_transformer_diagnostic.py cell first.")

TOLERANCE_DAYS = 10

def event_level_recall(flagged_dates, events, tolerance_days=TOLERANCE_DAYS):
    """Fraction of ground-truth EVENTS with at least one flagged day within
    tolerance_days of the event window. Denominator = number of events,
    fixed regardless of tolerance window size, so this is monotonic:
    widening tolerance can only detect more events, never fewer."""
    flagged_dates = pd.to_datetime(pd.Series(flagged_dates))
    detected = 0
    for start, end in events:
        start_ext = pd.Timestamp(start) - pd.Timedelta(days=tolerance_days)
        end_ext = pd.Timestamp(end) + pd.Timedelta(days=tolerance_days)
        hit = ((flagged_dates >= start_ext) & (flagged_dates <= end_ext)).any()
        if hit:
            detected += 1
    return detected / len(events)

def day_level_tolerance_precision(flagged_dates, events, tolerance_days=TOLERANCE_DAYS):
    """Precision doesn't have the denominator problem — denominator is
    'days flagged', which is fixed. A flagged day counts as correct if it
    falls within tolerance_days of ANY event window."""
    flagged_dates = pd.to_datetime(pd.Series(flagged_dates))
    if len(flagged_dates) == 0:
        return 0.0
    correct = np.zeros(len(flagged_dates), dtype=bool)
    for start, end in events:
        start_ext = pd.Timestamp(start) - pd.Timedelta(days=tolerance_days)
        end_ext = pd.Timestamp(end) + pd.Timedelta(days=tolerance_days)
        correct |= ((flagged_dates >= start_ext) & (flagged_dates <= end_ext)).values
    return correct.mean()

def f1_from_pr(p, r):
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)

# ---- Get the actual flagged DATES (not just 0/1 arrays) for each method ----
wti_dates_full = pd.to_datetime(wti_test["Date"].values)
wti_dates_seq  = pd.to_datetime(wti_test["Date"].values[SEQ_LEN:])
hh_dates_full  = pd.to_datetime(hh_test["Date"].values)
hh_dates_seq   = pd.to_datetime(hh_test["Date"].values[SEQ_LEN:])

combos = [
    ("WTI", "LSTM Autoencoder",    wti_dates_seq[wti_lstm_flag.astype(bool)],  wti_dates_seq),
    ("WTI", "Transformer Encoder", wti_dates_seq[wti_trans_flag.astype(bool)], wti_dates_seq),
    ("WTI", "Isolation Forest",    wti_dates_full[wti_iso_flag.astype(bool)],  wti_dates_full),
    ("WTI", "DBSCAN",              wti_dates_full[wti_db_flag.astype(bool)],   wti_dates_full),
    ("WTI", "Rolling Z-score",     wti_dates_full[wti_z_flag.astype(bool)],    wti_dates_full),
    ("Henry Hub", "LSTM Autoencoder",    hh_dates_seq[hh_lstm_flag.astype(bool)],  hh_dates_seq),
    ("Henry Hub", "Transformer Encoder", hh_dates_seq[hh_trans_flag.astype(bool)], hh_dates_seq),
    ("Henry Hub", "Isolation Forest",    hh_dates_full[hh_iso_flag.astype(bool)],  hh_dates_full),
    ("Henry Hub", "DBSCAN",              hh_dates_full[hh_db_flag.astype(bool)],   hh_dates_full),
    ("Henry Hub", "Rolling Z-score",     hh_dates_full[hh_z_flag.astype(bool)],    hh_dates_full),
]

rows = []
for commodity, method, flagged_dates, all_dates in combos:
    event_recall = event_level_recall(flagged_dates, GROUND_TRUTH_EVENTS)
    day_precision = day_level_tolerance_precision(flagged_dates, GROUND_TRUTH_EVENTS)
    f1 = f1_from_pr(day_precision, event_recall)
    n_events_detected = round(event_recall * len(GROUND_TRUTH_EVENTS))
    rows.append({
        "Commodity": commodity,
        "Method": method,
        "EventTol_Precision(day-level)": round(day_precision, 3),
        "EventTol_Recall(event-level)": round(event_recall, 3),
        "Events_Detected": f"{n_events_detected}/{len(GROUND_TRUTH_EVENTS)}",
        "EventTol_F1": round(f1, 3),
    })

event_tol_df = pd.DataFrame(rows)
print("=" * 90)
print(f"CORRECTED EVENT-LEVEL TOLERANCE METRIC (+/- {TOLERANCE_DAYS} days)")
print("Recall denominator = number of documented events (4) — fixed, so")
print("wider tolerance can only increase recall, never decrease it.")
print("=" * 90)
print(event_tol_df.to_string(index=False))

event_tol_df.to_csv("event_level_tolerance_results.csv", index=False)
print("\nSaved: event_level_tolerance_results.csv")

from google.colab import files
files.download("event_level_tolerance_results.csv")
