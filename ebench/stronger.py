"""Stronger and simpler comparison detectors.

Why these.  The benchmark literature (Wu & Keogh 2023; Schmidl et al. 2022; Paparrizos et al. 2022;
Liu & Paparrizos 2024) reports that simple and distance-based methods are often as good as deep
ones.  A deep model that wins here is therefore only informative if it also beats

  diff_z            Wu-Keogh style one-liner: |first difference| z-scored on the clean window
  seasonal_naive    |causal seasonal-naive residual| (needs --features resid): the feature the
                    neural models are given, used directly as a score.  If this matches the
                    LSTM-AE, the LSTM-AE's gain is the feature, not the model.
  window_pca        linear autoencoder on the SAME 30-step feature windows the neural models see
                    (TSB-AD "Sub-PCA")
  window_knn        k-NN distance on the same windows (TSB-AD "Sub-KNN"; k=1 is a matrix-profile
                    style AB-join on feature windows)
  window_iforest    isolation forest on the same windows (TSB-AD "Sub-IForest")
  matrix_profile    z-normalised 1-NN distance from each evaluation subsequence to the clean-window
                    subsequences = the AB-join matrix profile (Yeh et al. 2016), computed with
                    BLAS instead of stumpy.  Z-normalisation makes it blind to level and amplitude
                    (an invariance in the sense of Wu & Keogh); it is therefore a diagnostic of
                    which event types are level events.
  spectral_residual SR saliency (Ren et al. 2019; in TSB-AD), training-free, block-wise

All scorers share the signature of ebench.tuning (cfg, Ftr, Fev, cols, seed, spd=...) and return a
score per row of Fev (NaN where a window is not complete, as for the neural models).  They are
INDUCTIVE: every reference quantity comes from Ftr only.

Everything is numpy / scikit-learn; stumpy and TSB-AD are NOT required.  `experiments/crosscheck_official.py`
compares matrix_profile and VUS-PR with stumpy / the official TSB-AD package on your machine.
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from .windows import make_windows

SEQ_LEN = 30            # the neural models' window length (ebench.neural.reconstruction_scores)
MAX_REF = 8000          # reference windows kept for distance methods (uniform subsample, deterministic)


# ------------------------------------------------------------------ helpers
def _subsample(W, max_ref=MAX_REF):
    if len(W) <= max_ref:
        return W
    idx = np.linspace(0, len(W) - 1, max_ref).astype(int)
    return W[idx]


def _scatter(n, idx, vals):
    out = np.full(n, np.nan)
    if len(idx):
        out[idx] = vals
    return out


def _flat_windows(cfg, Ftr, Fev, cols, seq_len=SEQ_LEN):
    sc = StandardScaler().fit(Ftr[cols])
    Wtr, _ = make_windows(sc.transform(Ftr[cols]), Ftr.index, seq_len)
    Wev, idx = make_windows(sc.transform(Fev[cols]), Fev.index, seq_len)
    return Wtr.reshape(len(Wtr), -1), Wev.reshape(len(Wev), -1), idx


# ------------------------------------------------------------------ one-liner baselines
def score_diff_z(cfg, Ftr, Fev, cols, seed, **_):
    """|ret - mean| / sd, with mean and sd from the clean window (Wu & Keogh one-liner)."""
    r_tr = np.asarray(Ftr["ret"], float)
    mu, sd = np.nanmean(r_tr), np.nanstd(r_tr) + 1e-12
    return np.abs(np.asarray(Fev["ret"], float) - mu) / sd


def score_seasonal_naive(cfg, Ftr, Fev, cols, seed, **_):
    """|causal seasonal-naive residual|.  Requires the `resid` feature column."""
    if "resid" not in Fev.columns:
        raise ValueError("seasonal_naive needs --features resid")
    return np.abs(np.asarray(Fev["resid"], float))


# ------------------------------------------------------------------ windowed feature methods
def score_window_pca(cfg, Ftr, Fev, cols, seed, **_):
    Xtr, Xev, idx = _flat_windows(cfg, Ftr, Fev, cols)
    k = int(min(cfg["n_components"], Xtr.shape[1], len(Xtr) - 1))
    pca = PCA(n_components=k, random_state=seed).fit(Xtr)
    rec = pca.inverse_transform(pca.transform(Xev))
    return _scatter(len(Fev), idx, ((Xev - rec) ** 2).mean(1))


def score_window_knn(cfg, Ftr, Fev, cols, seed, **_):
    Xtr, Xev, idx = _flat_windows(cfg, Ftr, Fev, cols)
    nn = NearestNeighbors(n_neighbors=int(cfg["k"]), algorithm="brute").fit(_subsample(Xtr))
    d = nn.kneighbors(Xev)[0][:, -1] if len(Xev) else np.empty(0)
    return _scatter(len(Fev), idx, d)


def score_window_iforest(cfg, Ftr, Fev, cols, seed, **_):
    Xtr, Xev, idx = _flat_windows(cfg, Ftr, Fev, cols)
    m = IsolationForest(n_estimators=int(cfg["n_estimators"]), random_state=seed, n_jobs=1).fit(_subsample(Xtr, 4000))
    s = -m.score_samples(Xev) if len(Xev) else np.empty(0)
    return _scatter(len(Fev), idx, s)


# ------------------------------------------------------------------ matrix profile (AB-join)
def _znorm_windows(x, times, m):
    W, idx = make_windows(np.asarray(x, float).reshape(-1, 1), times, m)
    W = W[..., 0] if W.ndim == 3 else W
    mu = W.mean(1, keepdims=True)
    sd = W.std(1, keepdims=True)
    return (W - mu) / np.where(sd < 1e-8, 1.0, sd), sd[:, 0] < 1e-8, idx


def score_matrix_profile(cfg, Ftr, Fev, cols, seed, spd=48, **_):
    """AB-join matrix profile on the raw `value` column: for each evaluation subsequence of length
    m = window_days*spd, the Euclidean distance between z-normalised subsequences to its nearest
    clean-window subsequence.  d^2 = 2m(1 - corr), corr via one matrix product."""
    m = int(max(8, round(cfg["window_days"] * spd)))
    Ztr, _, _ = _znorm_windows(Ftr["value"], Ftr.index, m)
    Zev, flat, idx = _znorm_windows(Fev["value"], Fev.index, m)
    if len(Ztr) == 0 or len(Zev) == 0:
        return np.full(len(Fev), np.nan)
    ref = _subsample(Ztr)
    best = np.empty(len(Zev))
    for i in range(0, len(Zev), 2048):
        corr = Zev[i:i + 2048] @ ref.T / m
        best[i:i + 2048] = np.sqrt(np.maximum(2 * m * (1 - corr.max(1)), 0.0))
    best[flat] = 0.0                       # a constant subsequence has no shape to compare
    return _scatter(len(Fev), idx, best)


# ------------------------------------------------------------------ spectral residual
def spectral_residual_saliency(x, q=3):
    """Saliency map of Ren et al. (2019): inverse FFT of (log-amplitude minus its q-point mean)
    with the original phase."""
    x = np.asarray(x, float)
    f = np.fft.fft(x)
    amp, phase = np.abs(f), np.angle(f)
    la = np.log(amp + 1e-8)
    kernel = np.ones(q) / q
    avg = np.convolve(np.pad(la, (q // 2, q - 1 - q // 2), mode="edge"), kernel, mode="valid")
    return np.abs(np.fft.ifft(np.exp(la - avg + 1j * phase)))


def score_spectral_residual(cfg, Ftr, Fev, cols, seed, spd=48, **_):
    """Block-wise SR on `value`: blocks of block_days (50% overlap), score = (S - local mean S) /
    local mean S with a 21-point local mean, max over the blocks that cover a point.  Uses no
    training data (Ftr is ignored), so it is trivially inductive."""
    x = np.asarray(Fev["value"], float)
    L = int(max(64, round(cfg["block_days"] * spd)))
    n = len(x)
    out = np.zeros(n)
    if n < 8:
        return out
    L = min(L, n)
    for s in list(range(0, max(n - L, 0) + 1, max(L // 2, 1))) + ([n - L] if n > L else []):
        S = spectral_residual_saliency(x[s:s + L])
        loc = np.convolve(np.pad(S, (10, 10), mode="edge"), np.ones(21) / 21, mode="valid")
        sc = (S - loc) / (loc + 1e-12)
        out[s:s + L] = np.maximum(out[s:s + L], sc)
    return out


# ------------------------------------------------------------------ registry (merged into tuning.FAMILIES)
STRONGER = {
    "diff_z": dict(fn=score_diff_z, default={}, space={"_": [0]}, requires=[]),
    "seasonal_naive": dict(fn=score_seasonal_naive, default={}, space={"_": [0]}, requires=["resid"]),
    "window_pca": dict(fn=score_window_pca, default=dict(n_components=8),
                       space=dict(n_components=[2, 4, 8, 16, 32]), requires=[]),
    "window_knn": dict(fn=score_window_knn, default=dict(k=1), space=dict(k=[1, 3, 5, 10, 20]), requires=[]),
    "window_iforest": dict(fn=score_window_iforest, default=dict(n_estimators=100),
                           space=dict(n_estimators=[50, 100, 200, 400]), requires=[]),
    "matrix_profile": dict(fn=score_matrix_profile, default=dict(window_days=1.0),
                           space=dict(window_days=[0.25, 0.5, 1.0, 2.0, 7.0]), requires=[]),
    "spectral_residual": dict(fn=score_spectral_residual, default=dict(block_days=7.0),
                              space=dict(block_days=[1.0, 3.0, 7.0, 14.0, 28.0]), requires=[]),
}
