import numpy as np, pandas as pd
from ebench.crossseries import partial_spearman, stratified_partial_spearman, mean_rank_by_bin


def test_partial_spearman_removes_confounder():
    rng = np.random.default_rng(0)
    z = rng.normal(size=400)
    x = z + 0.3 * rng.normal(size=400)
    y = z + 0.3 * rng.normal(size=400)
    assert partial_spearman(x, y) > 0.7
    assert abs(partial_spearman(x, y, z)) < 0.2


def test_constant_input_is_nan():
    assert np.isnan(partial_spearman(np.ones(20), np.arange(20)))


def _frame(slope, n_ds=6, n=30, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_ds):
        x = rng.normal(size=n)
        rows.append(pd.DataFrame(dict(dataset=f"d{g}", x=x, y=slope * x + rng.normal(size=n), z=rng.normal(size=n))))
    return pd.concat(rows, ignore_index=True)


def test_stratified_detects_positive_negative_and_null():
    r, lo, hi, per = stratified_partial_spearman(_frame(1.5), "x", "y", "z", n_boot=200)
    assert lo > 0 and len(per) == 6
    r, lo, hi, _ = stratified_partial_spearman(_frame(-1.5), "x", "y", "z", n_boot=200)
    assert hi < 0
    r, lo, hi, _ = stratified_partial_spearman(_frame(0.0, seed=3), "x", "y", "z", n_boot=200)
    assert lo < 0 < hi


def test_small_groups_dropped_and_empty_is_nan():
    df = _frame(1.0, n_ds=3, n=5)
    r, lo, hi, per = stratified_partial_spearman(df, "x", "y", min_n=8)
    assert per.empty and np.isnan(r) and np.isnan(lo)


def test_mean_rank_by_bin():
    df = pd.DataFrame(dict(a=[.9, .8, .1, .2], b=[.1, .2, .9, .8], L=[1, 2, 10, 11]))
    t = mean_rank_by_bin(df, ["a", "b"], "{}", "L", [0, 5, 20])
    assert t.iloc[0]["a"] == 1 and t.iloc[1]["b"] == 1
