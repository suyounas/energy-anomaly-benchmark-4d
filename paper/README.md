# paper/

* `revised/` — the current manuscript (LaTeX, generic article class), generated tables and
  quoted numbers (`make_tables.py`, checked by `tests/test_paper_numbers.py`). **Use this.**
* The markdown files in this folder (`00_unified_title_abstract.md`, `section8`–`section11`,
  `draft_paper.md`) are the **pre-review draft**. They are kept for history. After peer review
  the following statements in them are **superseded or unsupported**:

| Pre-review statement | Where | Status |
|---|---|---|
| WTI DBSCAN F1 0.695 and "DBSCAN best on daily prices" | `00_unified_title_abstract.md`, `draft_paper.md`, `section9`, `section11` | Hybrid metric (day-level precision with event-level recall). Consistent same-day F1: DBSCAN 0.344, LSTM-AE 0.590. |
| "Detection gap" (no method above F1 0.25 at ~100–3,500 samples) and the sample-count selection rule | `section11_four_dataset_synthesis.md` | Withdrawn: GB LSTM 0.282 and FR line flow 0.408 lie inside the range; rule never tested on held-out events. |
| "DBSCAN declines monotonically with sample count" | `section11_four_dataset_synthesis.md` | Withdrawn: not reproduced in the constant-prevalence sweep. |
| "94% of LSTM flags land on the record summer of 2023" | `section10_ercot_us_demand.md`, `section11` | Came from a diagnostic script that is not in the repository; being re-run. Documented ERCOT 2023 records are treated as a post-hoc sensitivity tier. |
| "adapts the ATLAS statistical framework" | `draft_paper.md` | Removed: no such adaptation is described or implemented. |
| "Rolling z-score" in Studies 2–4 | all sections | In code it is `|return|` above its 95th test percentile (a fixed 5% alert budget), not a rolling z > 3. |

See `docs/REVISION_LOG.md` for the full status table and `paper/revised/main.tex` for the corrected text.
