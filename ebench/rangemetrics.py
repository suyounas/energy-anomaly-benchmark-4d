"""Range-aware, threshold-free metrics (Paparrizos et al. 2022 "VUS"; recommended by TSB-AD).

`range_auc_pr` and `vus_pr` are RE-IMPLEMENTED here from the published definition so the energy
experiments need no extra dependency.  They are not guaranteed to be numerically identical to
the official package: the buffer ramp and the integration rule are parameterised below.  Run
`experiments/crosscheck_official.py` on a machine with TSB-AD installed to quantify the agreement
(rank correlation across detectors is what the paper relies on) before quoting VUS values.

Definition used
  * Buffer.  Each labelled range [s, e] (inclusive) gets a soft label  l_t = sqrt(1 - d/(h+1))
    for the h = buffer//2 timestamps at distance d = 1..h before s and after e; labels inside the
    range are 1; l_t = max over overlapping ramps.
  * For a threshold theta (pred_t = score_t >= theta):
        precision(theta) = sum(l_t * pred_t) / sum(pred_t)
        recall(theta)    = sum(l_t * pred_t) / sum(l_t)  *  existence(theta)
    existence(theta) = fraction of labelled ranges whose buffered span contains at least one
    prediction (the "existence reward"), so flagging one long event cannot score full recall for
    a series with several events.
  * range_auc_pr = step-wise average precision over the recall axis (sklearn-style; the official
    code integrates trapezoidally -- a small, known difference).
  * vus_pr = mean of range_auc_pr over buffers 0, ..., max_buffer (n_buffers values).
"""
from __future__ import annotations
import numpy as np


def _runs(y):
    y = np.asarray(y).astype(int)
    d = np.diff(np.r_[0, y, 0])
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0] - 1))


def soft_labels(y, buffer):
    lab = np.asarray(y, float).copy()
    n, h = len(lab), int(buffer) // 2
    runs = _runs(y)
    if h > 0:
        d = np.arange(1, h + 1)
        w = np.sqrt(1 - d / (h + 1))
        for s, e in runs:
            for pos, ok in ((s - d, s - d >= 0), (e + d, e + d < n)):
                lab[pos[ok]] = np.maximum(lab[pos[ok]], w[ok])
    return lab, runs


def range_auc_pr(y_true, score, buffer=0):
    y = np.asarray(y_true).astype(int)
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    sc = np.asarray(score, float)
    fin = np.isfinite(sc)
    sc = np.where(fin, sc, (sc[fin].min() - 1.0) if fin.any() else 0.0)     # NaN scores rank last
    lab, runs = soft_labels(y, buffer)
    n, h = len(y), int(buffer) // 2
    order = np.argsort(-sc, kind="stable")
    ss = sc[order]
    last = np.r_[np.where(np.diff(ss) != 0)[0], n - 1]          # end of each tie group
    tp = np.cumsum(lab[order])[last]
    npred = (np.arange(n) + 1)[last]
    thr = ss[last]
    mx = np.sort([sc[max(s - h, 0): e + h + 1].max() for s, e in runs])
    exist = (len(mx) - np.searchsorted(mx, thr, side="left")) / len(mx)
    prec = tp / npred
    rec = tp / lab.sum() * exist
    rec_prev = np.r_[0.0, rec[:-1]]
    return float(np.sum(np.maximum(rec - rec_prev, 0.0) * prec))


def vus_pr(y_true, score, max_buffer=None, n_buffers=11):
    """Volume under the range-PR surface: mean range_auc_pr over buffers 0..max_buffer.
    max_buffer defaults to the median labelled-range length (clipped to [2, n//10])."""
    y = np.asarray(y_true).astype(int)
    runs = _runs(y)
    if not runs:
        return float("nan")
    if max_buffer is None:
        med = int(np.median([e - s + 1 for s, e in runs]))
        max_buffer = int(np.clip(med, 2, max(2, len(y) // 10)))
    bufs = np.unique(np.linspace(0, max_buffer, n_buffers).astype(int))
    return float(np.mean([range_auc_pr(y, score, b) for b in bufs]))
