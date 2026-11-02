"""Cross-check our re-implementations against the OFFICIAL packages.  Run on a machine where
`pip install stumpy TSB-AD` works (the authoring sandbox cannot reach PyPI).

  python3 experiments/crosscheck_official.py

1. matrix_profile (ebench.stronger) vs stumpy.stump AB-join: must agree to ~1e-4.
2. range_auc_pr / vus_pr (ebench.rangemetrics) vs TSB_AD VUS-PR on a battery of synthetic
   score/label pairs: reports Spearman rank correlation and the mean absolute difference.  They
   are NOT expected to be identical (ramp shape and integration rule differ); the paper may only
   rely on rank agreement, and must say which implementation produced a number.
Anything that fails prints what to paste back; nothing here modifies results.
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ebench import range_auc_pr, vus_pr
from ebench.stronger import score_matrix_profile


def check_matrix_profile():
    try:
        import stumpy
    except Exception as e:                                  # noqa: BLE001
        print("SKIP stumpy:", e); return
    rng = np.random.default_rng(0)
    n_tr, n_ev, m = 600, 400, 24
    x = np.cumsum(rng.normal(size=n_tr + n_ev)) + 3 * np.sin(np.arange(n_tr + n_ev) / 7)
    t = pd.date_range("2020-01-01", periods=len(x), freq="h")
    Ftr = pd.DataFrame({"value": x[:n_tr]}, index=t[:n_tr]); Fev = pd.DataFrame({"value": x[n_tr:]}, index=t[n_tr:])
    ours = score_matrix_profile(dict(window_days=m / 24), Ftr, Fev, ["value"], 0, spd=24)
    theirs = stumpy.stump(Fev["value"].to_numpy(float), m, Ftr["value"].to_numpy(float), ignore_trivial=False)[:, 0].astype(float)
    # ours is indexed by window END, stumpy by window START
    a, b = ours[m - 1:], theirs
    d = np.nanmax(np.abs(a - b))
    print(f"matrix_profile vs stumpy AB-join: max |diff| = {d:.2e}  ->", "OK" if d < 1e-3 else "MISMATCH (paste this)")


def check_vus():
    try:
        from TSB_AD.evaluation.metrics import get_metrics
    except Exception as e:                                  # noqa: BLE001
        print("SKIP TSB-AD:", e); return
    rng = np.random.default_rng(1)
    rows = []
    for case in range(40):
        n = 3000
        y = np.zeros(n, int)
        for _ in range(rng.integers(1, 5)):
            L = int(rng.integers(5, 120)); s = int(rng.integers(0, n - L)); y[s:s + L] = 1
        snr = rng.uniform(0.2, 3.0); lag = int(rng.integers(0, 15))
        score = np.roll(y, lag) * snr + rng.normal(size=n)
        run = np.diff(np.r_[0, y, 0]); med = int(np.median(np.where(run == -1)[0] - np.where(run == 1)[0]))
        win = max(2, med)
        try:
            off = get_metrics(score, y, metric="vus", slidingWindow=win)
            off_v = off.get("VUS-PR", off.get("VUS_PR"))
        except Exception as e:                              # noqa: BLE001
            print("official call failed; paste this error:", repr(e)); return
        rows.append((vus_pr(y, score, max_buffer=win), off_v))
    r = pd.DataFrame(rows, columns=["ours", "official"])
    print(f"VUS-PR ours vs official: Spearman = {r.corr(method='spearman').iloc[0,1]:.3f}, "
          f"mean |diff| = {(r.ours - r.official).abs().mean():.3f}  (n={len(r)})")
    print("acceptance: Spearman >= 0.95 is enough to rank detectors; quote official numbers where available")


if __name__ == "__main__":
    check_matrix_profile(); check_vus()
