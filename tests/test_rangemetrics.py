"""Properties of the range-aware metrics (not a numerical match to the official package)."""
import numpy as np
import pytest

from ebench import range_auc_pr, vus_pr, auc_pr
from ebench.rangemetrics import soft_labels, _runs


def two_events(n=1000):
    y = np.zeros(n, int); y[100:140] = 1; y[600:660] = 1
    return y


def test_runs_are_inclusive():
    assert _runs(np.array([0, 1, 1, 0, 1])) == [(1, 2), (4, 4)]


def test_soft_labels_ramp_is_in_unit_interval_and_symmetric():
    y = np.zeros(50, int); y[20:25] = 1
    lab, _ = soft_labels(y, 8)
    assert lab.max() == 1.0 and lab[:16].sum() == 0 and lab[29:].sum() == 0
    assert lab[19] == pytest.approx(lab[25]) and 0 < lab[17] < lab[19] < 1


def test_perfect_score_is_one_and_inverted_score_is_low():
    y = two_events()
    assert range_auc_pr(y, y.astype(float)) == pytest.approx(1.0)
    # a binary score cannot cover the soft ramps, so VUS < 1 by construction (as in the official
    # measure); a score equal to the soft labels reaches ~1 at that buffer
    lab, _ = soft_labels(y, 10)
    assert range_auc_pr(y, lab, 10) > 0.99
    assert 0.7 < vus_pr(y, y.astype(float)) < 1.0
    assert range_auc_pr(y, 1.0 - y) <= y.mean() + 0.01


def test_random_score_is_near_prevalence():
    y = two_events(); rng = np.random.default_rng(0)
    v = np.mean([range_auc_pr(y, rng.random(len(y))) for _ in range(30)])
    assert v < 3 * y.mean()


def test_existence_reward_penalises_missing_an_event():
    y = two_events()
    only_first = np.where((np.arange(len(y)) >= 100) & (np.arange(len(y)) < 140), 1.0, 0.0)
    both = y.astype(float)
    assert range_auc_pr(y, only_first) < range_auc_pr(y, both) - 0.3


def test_buffer_gives_credit_for_near_misses():
    """A detector firing 12 steps early (no overlap with a 10-step event) scores ~chance at buffer 0
    and clearly more once the buffer reaches it."""
    y = np.zeros(1000, int); y[100:110] = 1; y[600:610] = 1
    s = np.roll(y.astype(float), -12)
    assert range_auc_pr(y, s, 0) < 0.1
    assert range_auc_pr(y, s, 40) > 3 * range_auc_pr(y, s, 0)
    assert vus_pr(y, s, max_buffer=40) > range_auc_pr(y, s, 0)


def test_nan_scores_rank_last_and_nan_only_for_degenerate_labels():
    y = two_events(); s = y.astype(float); s[:5] = np.nan
    assert range_auc_pr(y, s) == pytest.approx(1.0)
    assert np.isnan(range_auc_pr(np.zeros(10, int), np.arange(10.)))
    assert np.isnan(vus_pr(np.zeros(10, int), np.arange(10.)))


def test_ties_do_not_inflate():
    y = two_events()
    assert range_auc_pr(y, np.zeros(len(y))) == pytest.approx(y.mean(), abs=0.05)
