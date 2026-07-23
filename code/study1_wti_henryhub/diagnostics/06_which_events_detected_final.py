# Quick check: which SPECIFIC event(s) did each method detect within tolerance?
# Also restricts the denominator to only the 2 events actually reachable
# given the train/test split (Texas Freeze, Russia-Ukraine).

DETECTABLE_EVENTS = [
    ("2021-02-10", "2021-02-25"),  # Texas deep freeze
    ("2022-02-20", "2022-06-30"),  # Russia-Ukraine energy crisis
]
EVENT_NAMES = {0: "Texas Freeze 2021", 1: "Russia-Ukraine 2022"}

def which_events_detected(flagged_dates, events=DETECTABLE_EVENTS, tolerance_days=10):
    flagged_dates = pd.to_datetime(pd.Series(flagged_dates))
    hits = []
    for i, (start, end) in enumerate(events):
        start_ext = pd.Timestamp(start) - pd.Timedelta(days=tolerance_days)
        end_ext = pd.Timestamp(end) + pd.Timedelta(days=tolerance_days)
        matching = flagged_dates[(flagged_dates >= start_ext) & (flagged_dates <= end_ext)]
        if len(matching) > 0:
            first_hit = matching.min()
            lag_days = (first_hit - pd.Timestamp(start)).days
            hits.append((EVENT_NAMES[i], True, first_hit.date(), lag_days))
        else:
            hits.append((EVENT_NAMES[i], False, None, None))
    return hits

print("=" * 80)
print("WHICH SPECIFIC EVENT DID EACH METHOD DETECT? (of the 2 reachable events)")
print("=" * 80)

for commodity, method, flagged_dates, all_dates in combos:
    print(f"\n{commodity} — {method}")
    for name, detected, first_date, lag in which_events_detected(flagged_dates):
        if detected:
            print(f"  [DETECTED] {name}: first flagged {first_date} "
                  f"({'+' if lag >= 0 else ''}{lag} days from event start)")
        else:
            print(f"  [MISSED]   {name}")

# Corrected recall using the true denominator of 2 reachable events
print("\n" + "=" * 80)
print("CORRECTED EVENT-LEVEL RECALL — denominator = 2 reachable events only")
print("=" * 80)
rows_corrected = []
for commodity, method, flagged_dates, all_dates in combos:
    hits = which_events_detected(flagged_dates)
    n_detected = sum(1 for h in hits if h[1])
    day_precision = day_level_tolerance_precision(flagged_dates, GROUND_TRUTH_EVENTS)
    recall_corrected = n_detected / len(DETECTABLE_EVENTS)
    f1_corrected = f1_from_pr(day_precision, recall_corrected)
    rows_corrected.append({
        "Commodity": commodity, "Method": method,
        "Precision": round(day_precision, 3),
        "Recall(of 2 reachable)": f"{n_detected}/2 = {round(recall_corrected, 2)}",
        "F1_corrected": round(f1_corrected, 3),
    })

corrected_df = pd.DataFrame(rows_corrected)
print(corrected_df.to_string(index=False))
corrected_df.to_csv("final_corrected_results.csv", index=False)
print("\nSaved: final_corrected_results.csv")

from google.colab import files
files.download("final_corrected_results.csv")
