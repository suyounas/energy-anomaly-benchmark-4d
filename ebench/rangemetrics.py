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
  * `segments` (optional id per row): ramps, ranges and the existence reward never cross a segment
    boundary.  Used by the bootstrap, whose replicates glue unrelated stretches of the series together.
"""
from __future__ import annotations
import numpy as np


def _runs(y):
    y = np.asarray(y).astype(int)
    d = np.diff(np.r_[0, y, 0])
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0] - 1))


def _segment_bounds(n, segments):
    """[(start, stop)) of the contiguous stretches of equal segment id (the whole series if None)."""
    if segments is None:
        return [(0, n)]
    seg = np.asarray(segments)
    cut = np.r_[0, np.where(seg[1:] != seg[:-1])[0] + 1, n]
    return list(zip(cut[:-1], cut[1:]))


def _labels_and_ranges(y, buffer, segments=None):
    """Soft labels and, per labelled range, (s, e, lo, hi): the range and the span its buffer may touch.
    With `segments` (an id per row) ramps and ranges never cross a segment boundary, so rows that were
    glued together by a resampling procedure do not influence each other."""
    y = np.asarray(y).astype(int)
    lab = y.astype(float)
    h = int(buffer) // 2
    ranges = []
    for a, b in _segment_bounds(len(y), segments):
        for s, e in _runs(y[a:b]):
            s, e = s + a, e + a
            lo, hi = max(s - h, a), min(e + h, b - 1)
            ranges.append((s, e, lo, hi))
            if h > 0:
                d = np.arange(1, h + 1)
                w = np.sqrt(1 - d / (h + 1))
                for pos in (s - d, e + d):
                    ok = (pos >= a) & (pos < b)
                    lab[pos[ok]] = np.maximum(lab[pos[ok]], w[ok])
    return lab, ranges


def soft_labels(y, buffer, segments=None):
    lab, ranges = _labels_and_ranges(y, buffer, segments)
    return lab, [(s, e) for s, e, _, _ in ranges]


def range_auc_pr(y_true, score, buffer=0, segments=None):
    y = np.asarray(y_true).astype(int)
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    sc = np.asarray(score, float)
    fin = np.isfinite(sc)
    sc = np.where(fin, sc, (sc[fin].min() - 1.0) if fin.any() else 0.0)     # NaN scores rank last
    lab, ranges = _labels_and_ranges(y, buffer, segments)
    n = len(y)
    order = np.argsort(-sc, kind="stable")
    ss = sc[order]
    last = np.r_[np.where(np.diff(ss) != 0)[0], n - 1]          # end of each tie group
    tp = np.cumsum(lab[order])[last]
    npred = (np.arange(n) + 1)[last]
    thr = ss[last]
    mx = np.sort([sc[lo: hi + 1].max() for _, _, lo, hi in ranges])
    exist = (len(mx) - np.searchsorted(mx, thr, side="left")) / len(mx)
    prec = tp / npred
    rec = tp / lab.sum() * exist
    rec_prev = np.r_[0.0, rec[:-1]]
    return float(np.sum(np.maximum(rec - rec_prev, 0.0) * prec))


def vus_pr(y_true, score, max_buffer=None, n_buffers=11, segments=None):
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
    return float(np.mean([range_auc_pr(y, score, b, segments) for b in bufs]))
