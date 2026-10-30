"""Seasonal-naive residual feature: causal, zero on periodic data, sees level shifts."""
import numpy as np
import pandas as pd
import pytest

from real_resolution_ablation import build_features, seasonal_residual, FEATS, FEATS_RESID, feature_cols

SPD = 24


def _weekly(days=120, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    t = pd.date_range("2020-01-01", periods=days * SPD, freq="h")
    k = np.arange(len(t))
    x = 100 + 10 * np.sin(2 * np.pi * k / SPD) + 5 * np.sin(2 * np.pi * k / (7 * SPD)) + rng.normal(0, noise, len(t))
    return t, x


def test_resid_is_zero_on_a_pure_weekly_pattern():
    t, x = _weekly()
    r = seasonal_residual(pd.Series(x, index=t), SPD)
    assert np.nanmax(np.abs(r[4 * 7 * SPD:])) < 1e-6


def test_resid_exposes_a_level_shift_that_the_recipe_features_barely_see():
    t, x = _weekly(noise=1.0)
    x2 = x.copy(); s0, e0 = 90 * SPD, 93 * SPD
    x2[s0:e0] += 12.0
    r = pd.Series(seasonal_residual(pd.Series(x2, index=t), SPD), index=t)
    inside = r.iloc[s0 + SPD:e0 - SPD].abs().median()               # interior of the event, away from the edges
    outside = r.iloc[40 * SPD:80 * SPD].abs().median()
    assert inside > 5 * max(outside, 1e-9)
    F = build_features(t, x2, SPD, 30.0, with_resid=True)
    assert "resid" in F and all(c in F for c in FEATS_RESID)
    assert F.loc[t[s0 + SPD]:t[e0 - SPD], "ret"].abs().median() < 0.1      # ret is not level-aware


def test_resid_is_causal():
    t, x = _weekly(noise=1.0)
    a = seasonal_residual(pd.Series(x, index=t), SPD)
    x2 = x.copy(); x2[100 * SPD:] += 50.0                         # change only the future
    b = seasonal_residual(pd.Series(x2, index=t), SPD)
    cut = 100 * SPD
    # the scale (MAD) uses the whole input, so compare standardised shapes: ratio must be a constant
    m = np.isfinite(a[:cut]) & np.isfinite(b[:cut]) & (np.abs(a[:cut]) > 1e-9)
    ratio = b[:cut][m] / a[:cut][m]
    assert np.allclose(ratio, ratio[0], rtol=1e-6)


def test_feature_cols_and_default_keep_the_published_recipe():
    assert feature_cols("recipe") == FEATS and "resid" not in FEATS
    assert feature_cols("resid") == FEATS + ["resid"]
    t, x = _weekly()
    assert "resid" not in build_features(t, x, SPD, 30.0)
