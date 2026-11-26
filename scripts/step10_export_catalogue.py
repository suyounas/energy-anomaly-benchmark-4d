#!/usr/bin/env python3
"""Export the analysed Step 10 event catalogue.

Joins results/step10/runs_classical/events_final.csv (authority-events with
storm group, class, family, dates) with the source URL and mapping rationale
from results/step10/events_selected.csv (Class B) and docs/step10_events.csv
(Class A), and keeps the authorities that were actually run (45).

Output: results/step10/events_analysed.csv (1,973 authority-events).
No new information is created; the file only makes the catalogue self-contained.
"""
import csv
import glob
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
S10 = os.path.join(ROOT, "results", "step10")

ran = sorted({os.path.basename(f).split("_events")[0]
              for f in glob.glob(os.path.join(S10, "runs_classical", "*_events.csv"))})

src = {}
for path in (os.path.join(S10, "events_selected.csv"),
             os.path.join(ROOT, "docs", "step10_events.csv")):
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            src[r["event_id"]] = r

out_path = os.path.join(S10, "events_analysed.csv")
n = 0
with open(os.path.join(S10, "runs_classical", "events_final.csv"), newline="", encoding="utf-8") as fh, \
        open(out_path, "w", newline="", encoding="utf-8") as out:
    w = csv.writer(out)
    w.writerow(["ba", "event_id", "storm_group", "source_class", "family",
                "start", "end", "source_url", "mapping_rationale"])
    for r in csv.DictReader(fh):
        if r["ba"] not in ran:
            continue
        s = src.get(r["event_id"], {})
        w.writerow([r["ba"], r["event_id"], r["storm_group"], r["source_class"],
                    r["family"], r["start"], r["end"],
                    s.get("source_url", ""), s.get("mapping_rationale", "")])
        n += 1

print(f"wrote {out_path}: {n} authority-events, {len(ran)} authorities")
assert n == 1973, n
