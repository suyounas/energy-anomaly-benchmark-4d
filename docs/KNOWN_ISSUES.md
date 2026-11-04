# Known issues in earlier releases

These are defects found in earlier versions of the evaluation code. They are fixed in the current code and covered by regression tests in `tests/`.

1. **Event end day dropped.** The first evaluation code treated an event's end date as exclusive, so the last calendar day of every labelled event was scored as normal. For a 5-day event this removed one fifth of its labelled timestamps. Events are now inclusive calendar dates (`ebench/metrics.py::event_mask`).
2. **Windows across the training gap.** Neural windows in the clean-window split were built across the gap between the two training segments (about 30 windows). Current equal-budget runs use segment-aware windows (`ebench/windows.py`). The neural numbers of the five-detector pipeline (`results/study*`) were not regenerated.
3. **VUS-PR bootstrap.** A plain block bootstrap breaks events and lets buffers cross seams, which invalidates intervals for range metrics. Intervals computed this way were discarded; only point values are reported.
4. **Period estimator.** A flat training prefix broke the autocorrelation period estimator in the public-benchmark test; 40 series were skipped in a first run. The primary result includes them.
