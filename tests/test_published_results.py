"""Consistency checks on the published result tables in results/.

These do not re-run the models.  They check that reported numbers are
internally consistent (F1 really is the harmonic mean of the reported precision
and recall, everything is in [0, 1]) so a transcription or rounding slip is
caught before it reaches the paper.
"""
import glob
import os

import pandas as pd
import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..", "results")
TABLES = [
    "study1_wti_henryhub/strict_same_day_results.csv",
    "study2_grid_network/phase3_final_results.csv",
    "study3_gb_neso/neso_phase3_results.csv",
    "study4_ercot_eia/ercot_phase3_results.csv",
]


@pytest.mark.parametrize("rel", TABLES)
def test_f1_is_harmonic_mean_of_precision_and_recall(rel):
    df = pd.read_csv(os.path.join(ROOT, rel))
    p, r = df["Precision"], df["Recall"]
    expected = (2 * p * r / (p + r)).where((p + r) > 0, 0.0)
    # Each input is rounded to 3 dp, so allow a small tolerance.
    bad = df[(expected - df["F1"]).abs() > 0.004]
    assert bad.empty, f"{rel}: F1 inconsistent with P/R in rows\n{bad}"


@pytest.mark.parametrize("rel", TABLES)
def test_metrics_are_probabilities(rel):
    df = pd.read_csv(os.path.join(ROOT, rel))
    for col in ("Precision", "Recall", "F1"):
        assert df[col].between(0, 1).all(), f"{rel}: {col} outside [0, 1]"


@pytest.mark.parametrize("rel", TABLES)
def test_every_dataset_has_all_five_methods(rel):
    df = pd.read_csv(os.path.join(ROOT, rel))
    group_cols = [c for c in ("Commodity", "Dataset", "Protocol") if c in df.columns]
    for key, g in df.groupby(group_cols):
        assert g["Method"].nunique() == 5, f"{rel} {key}: methods {sorted(g['Method'])}"


def test_all_result_files_are_covered_or_deliberately_skipped():
    on_disk = {os.path.relpath(p, ROOT) for p in glob.glob(os.path.join(ROOT, "*", "*.csv"))}
    skipped = {"study1_wti_henryhub/final_corrected_results.csv",   # recall stored as text "1/2 = 0.5"
               "study1_wti_henryhub/which_events_detected.csv"}      # event table, no P/R/F1
    assert on_disk == set(TABLES) | skipped
