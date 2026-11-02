"""ebench: the single evaluation contract used by every study in this repository.

Rules enforced here:
  * Event-level and timestamp-level metrics are SEPARATE functions and are never mixed:
    event precision is always paired with event recall, timestamp precision with timestamp recall.
  * Event end dates are inclusive of the whole calendar day.
  * Per-event numbers are reported per event; pooled numbers are labelled as pooled.
  * A threshold-free metric (average precision / AUC-PR) is always available next to
    any fixed-budget F1, together with the F1 ceiling implied by prevalence.
"""
from .metrics import (  # noqa: F401
    event_end_exclusive, event_mask, per_event_coverage, event_level_prf,
    timestamp_level_prf, auc_pr, f1_ceiling_at_budget, block_bootstrap_ci,
    paired_block_bootstrap, summarise_paired,
)
from .windows import contiguous_segments, make_windows  # noqa: F401
from .rangemetrics import range_auc_pr, vus_pr  # noqa: F401
from .report import coverage_table, metric_intervals  # noqa: F401
