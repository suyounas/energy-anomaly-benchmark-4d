"""Tests for the taxonomy validation suite and macro-lift selection (ebench.tuning)."""
import numpy as np
import pandas as pd
import pytest

import ebench.tuning as tu
from ebench.synthetic import base_series, STEPS_PER_DAY
from ebench.tuning import make_validation_suite, macro_lift, suite_cells, equal_budget_search


def _series(days=400, seed=0):
    rng = np.random.default_rng(seed)
    t, x = base_series(days * STEPS_PER_DAY, rng)
    return t, x


def test_suite_cells_cover_the_declared_taxonomy():
    tbl = suite_cells(durations_days=(1, 3), magnitudes=(1.5,))
    assert set(tbl.type) == {"shift", "burst", "ramp", "pulse"}
    assert (tbl[tbl.type == "pulse"].duration_days == 0).all()
    assert len(tbl) == 3 * 2 + 1 and list(tbl.cell) == list(range(len(tbl)))


def test_suite_events_only_in_validation_region_and_never_overlap():
    t, x = _series()
    vs = 250 * STEPS_PER_DAY
    x2, cell, tbl = make_validation_suite(t, x, vs, durations_days=(1, 3))
    assert (cell[:vs] == -1).all() and np.array_equal(x2[:vs], x[:vs])        # fit part untouched
    assert set(np.unique(cell[cell >= 0])) == set(tbl.cell)                   # every cell has events
    changed = x2 != x
    assert (changed[cell < 0] == False).all()                                 # clean rows are untouched
    # >= 1 clean day between consecutive event runs
    lab = (cell >= 0).astype(int); edges = np.flatnonzero(np.diff(lab) != 0)
    runs = np.split(np.arange(len(lab)), edges + 1)
    gaps = [len(r) for r in runs if lab[r[0]] == 0][1:-1]
    assert min(gaps) >= STEPS_PER_DAY - 1


def test_suite_is_deterministic_and_seed_dependent():
    t, x = _series()
    vs = 250 * STEPS_PER_DAY
    a = make_validation_suite(t, x, vs, durations_days=(1, 3), seed=1)[1]
    b = make_validation_suite(t, x, vs, durations_days=(1, 3), seed=1)[1]
    c = make_validation_suite(t, x, vs, durations_days=(1, 3), seed=2)[1]
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_suite_refuses_a_validation_region_that_is_too_short():
    t, x = _series(days=270)
    with pytest.raises(ValueError, match="too short"):
        make_validation_suite(t, x, 250 * STEPS_PER_DAY, durations_days=(1, 3, 7))


def test_macro_lift_oracle_is_high_and_random_is_about_one():
    rng = np.random.default_rng(0)
    cell = np.full(6000, -1); cell[500:560] = 0; cell[2000:2100] = 1; cell[4000:4030] = 2
    oracle = (cell >= 0).astype(float) + 0.01 * rng.random(6000)
    ml, per = macro_lift(oracle, cell, 3)
    assert ml > 20 and all(p > 20 for p in per)
    ml_r, _ = macro_lift(rng.random(6000), cell, 3)
    assert 0.5 < ml_r < 2.0


def test_macro_lift_ignores_other_cells_events_as_negatives():
    cell = np.full(3000, -1); cell[100:150] = 0; cell[1500:1550] = 1
    s = np.zeros(3000); s[1500:1550] = 10.0; s[100:150] = 5.0             # cell 1 louder than cell 0
    _, per = macro_lift(s, cell, 2)
    assert per[0] == pytest.approx(1 / (50 / (50 + 2900)), rel=1e-6)       # perfect vs clean rows only


def test_macro_lift_handles_nan_scores():
    cell = np.full(2000, -1); cell[200:240] = 0
    s = (cell >= 0).astype(float); s[:30] = np.nan
    ml, per = macro_lift(s, cell, 1)
    assert np.isfinite(ml) and per[0] > 10


def _fake_search(monkeypatch, strength_fn, cells):
    """One fake family whose score is strength(cfg) * oracle + noise."""
    n = len(cells); rng = np.random.default_rng(0); noise = rng.random(n)
    oracle = (cells >= 0).astype(float)
    fam = dict(fn=lambda cfg, Ftr, Fev, cols, seed, **_: strength_fn(cfg) * oracle + noise,
               default=dict(q=0.0), space=dict(q=[0.5, 1.0, 2.0, 4.0]))
    monkeypatch.setitem(tu.FAMILIES, "fake", fam)
    F = pd.DataFrame({"a": np.zeros(n + 10)})
    y = (cells >= 0).astype(int)
    return equal_budget_search(["fake"], F.iloc[:10], F.iloc[10:], y, ["a"], n_trials=4, seed=0,
                               cells=cells, n_cells=int(cells.max()) + 1, min_spread=0.1)


def test_search_with_cells_selects_by_macro_lift(monkeypatch):
    cells = np.full(6000, -1); cells[500:560] = 0; cells[2000:2100] = 1; cells[4000:4030] = 2
    trials, chosen = _fake_search(monkeypatch, lambda c: c["q"], cells)
    assert trials.informative.all()
    assert chosen["fake"]["q"] == trials.loc[trials.val_macro_lift.idxmax(), "cfg_q"]
    assert chosen["fake"]["q"] >= 2.0                       # stronger signal wins
    assert any(c.startswith("lift_cell_") for c in trials.columns)


def test_search_with_cells_keeps_default_when_family_is_blind(monkeypatch):
    cells = np.full(6000, -1); cells[500:560] = 0; cells[2000:2100] = 1; cells[4000:4030] = 2
    trials, chosen = _fake_search(monkeypatch, lambda c: 0.0, cells)      # score = pure noise
    assert not trials.informative.any()
    assert chosen["fake"] == dict(q=0.0)
    assert trials.val_macro_lift.between(0.5, 2.0).all()


def test_legacy_search_path_is_unchanged_without_cells():
    t, x = _series(days=120)
    vs = 80 * STEPS_PER_DAY
    from ebench.tuning import inject_into_series
    from ebench.detectors import features
    x2, y = inject_into_series(t, x, vs, (1, 3), 0.1, 3.0, seed=0)
    F = features(x2, STEPS_PER_DAY)
    trials, chosen = equal_budget_search(["rolling_z"], F.iloc[:vs], F.iloc[vs:], y[vs:],
                                         ["value", "ret", "roll_mean", "roll_std"], n_trials=3)
    assert "val_macro_lift" not in trials.columns and "informative" in trials.columns
