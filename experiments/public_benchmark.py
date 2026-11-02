"""Step 8: a held-out test of the withdrawn duration rule on public labelled series (TSB-AD-U).

Pre-registered in docs/STEP8_PREREGISTRATION.md (read it before running).  CPU only.

  # 1. get the data on YOUR machine (the authoring sandbox cannot reach it):
  #    https://www.thedatum.org/datasets/TSB-AD-U.zip   (per-source licences differ; do not redistribute)
  # 2. run (resumable; one row per series):
  OMP_NUM_THREADS=1 python3 experiments/public_benchmark.py run --data TSB-AD-U --out results/public/tsbad_u.csv --jobs 4
  # 3. analyse:
  python3 experiments/public_benchmark.py analyse --results results/public/tsbad_u.csv --out docs/step8_report.md

Per series: train = the prefix named in the file name (`..._tr_<N>_...`, else --train-frac), test = the rest.  All
detectors are inductive (fitted on the prefix only) and scored on the SAME test rows.  The series has no
calendar, so a pseudo one-minute index is used and every time-based window is expressed in samples through the
estimated period P (autocorrelation peak on the prefix; a fallback is used and flagged when there is none).
"""
import argparse, glob, os, re, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ebench import auc_pr, vus_pr
from ebench.rangemetrics import _runs
from ebench.tuning import FAMILIES
from ebench import stratified_partial_spearman, mean_rank_by_bin

SPD = 1440                                   # pseudo-minute index: 1440 samples = "one day"
FEATS_PUBLIC = ["value", "ret", "roll_mean", "roll_std", "resid"]
DETECTORS = ["diff_z", "rolling_z", "seasonal_naive", "isolation_forest", "knn_inductive", "window_pca",
             "window_iforest", "matrix_profile", "spectral_residual"]
PRIMARY = ("window_pca", "knn_inductive")     # (reconstruction, density): d = auc_pr(A) - auc_pr(B)
MIN_TEST_ROWS = 100


# ------------------------------------------------------------------ loading
def parse_name(path):
    base = os.path.splitext(os.path.basename(path))[0]
    m = re.match(r"^(\d+)_([^_]+)_id_(\d+)", base)
    dataset = m.group(2) if m else (base.split("_")[1] if "_" in base else "unknown")
    tr = re.search(r"_tr_(\d+)", base)
    return dataset, (int(tr.group(1)) if tr else None)


def load_series(path):
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    lab = cols.get("label") or cols.get("labels") or cols.get("anomaly") or df.columns[-1]
    val = cols.get("data") or cols.get("value") or next(c for c in df.columns if c != lab)
    x = pd.to_numeric(df[val], errors="coerce").to_numpy(float)
    y = (pd.to_numeric(df[lab], errors="coerce").fillna(0).to_numpy() > 0).astype(int)
    if not np.isfinite(x).all():                       # forward-fill isolated gaps, then give up on the rest
        x = pd.Series(x).ffill().bfill().to_numpy(float)
    return x, y


def estimate_period(x, lo=4, hi=200, min_acf=0.2):
    """Autocorrelation peak of the (linearly detrended) prefix; returns (P, found)."""
    n = len(x)
    hi = int(min(hi, n // 3))
    if hi <= lo + 2:
        return max(lo, min(16, n // 4)), False
    t = np.arange(n)
    r = x - np.polyval(np.polyfit(t, x, 1), t)
    r = r - r.mean()
    f = np.fft.rfft(r, 2 * n)
    acf = np.fft.irfft(f * np.conj(f))[:n]
    acf = acf / (acf[0] + 1e-12)
    seg = acf[lo - 1: hi + 1]                       # local maxima only: a smooth series has a high ACF at tiny lags
    pk = [lo + i for i in range(1, len(seg) - 1) if seg[i] > seg[i - 1] and seg[i] >= seg[i + 1]]
    if pk:
        top = max(acf[k] for k in pk)
        k = next(k for k in pk if acf[k] >= 0.9 * top)   # smallest peak that is near the best (avoid multiples)
        if acf[k] >= min_acf:
            return k, True
    return int(np.clip(n // 50, 16, 100)), False


def public_features(x, P, n_train):
    n = len(x)
    idx = pd.date_range("2000-01-01", periods=n, freq="min")
    s = pd.Series(x, index=idx)
    w = max(8, int(P))
    F = pd.DataFrame({"value": x, "ret": s.diff().fillna(0.0).to_numpy(),
                      "roll_mean": s.rolling(w, min_periods=2).mean().bfill().to_numpy(),
                      "roll_std": s.rolling(w, min_periods=2).std().bfill().fillna(0.0).to_numpy()}, index=idx)
    past = np.column_stack([s.shift(k * P).to_numpy() for k in range(1, 5)])
    have = np.isfinite(past).sum(1) >= 1
    base = np.zeros(n)
    base[have] = np.nanmedian(past[have], axis=1)
    r = np.where(have, x - base, 0.0)
    mad = np.median(np.abs(r[:n_train] - np.median(r[:n_train]))) * 1.4826
    floor = 1e-3 * (np.std(x[:n_train]) or 1.0)
    F["resid"] = r / (mad if mad > floor else floor)
    return F


def detector_cfgs(P):
    d = P / SPD
    return {"diff_z": {}, "rolling_z": dict(window_days=5 * d), "seasonal_naive": {},
            "isolation_forest": dict(n_estimators=100, max_samples="auto"), "knn_inductive": dict(k=5),
            "window_pca": dict(n_components=8), "window_iforest": dict(n_estimators=100),
            "matrix_profile": dict(window_days=d), "spectral_residual": dict(block_days=max(64, 8 * P) / SPD)}


# ------------------------------------------------------------------ one series
def evaluate_file(path, train_frac=0.1, seed=0, detectors=DETECTORS):
    t0 = time.time()
    dataset, tr = parse_name(path)
    row = dict(file=os.path.basename(path), dataset=dataset, status="ok")
    try:
        x, y = load_series(path)
        n = len(x)
        tr = int(tr) if tr else int(train_frac * n)
        row.update(n=n, n_train=tr, train_contaminated=int(y[:tr].sum() > 0))
        if tr < 60 or n - tr < MIN_TEST_ROWS:
            return {**row, "status": "too_short"}
        P, found = estimate_period(x[:tr])
        row.update(period=P, period_found=int(found))
        F = public_features(x, P, tr)
        Ftr, Fev, yt = F.iloc[:tr], F.iloc[tr:], y[tr:]
        cfgs = detector_cfgs(P)
        scores, errs = {}, {}
        for d in detectors:
            try:
                scores[d] = np.asarray(FAMILIES[d]["fn"](cfgs[d], Ftr, Fev, FEATS_PUBLIC, seed, spd=SPD), float)
            except Exception as e:                                   # noqa: BLE001  (counted, never silent)
                errs[d] = f"{type(e).__name__}: {str(e)[:80]}"
        for d, e in errs.items():
            row[f"err_{d}"] = e
        if any(d in errs for d in PRIMARY):
            return {**row, "status": "primary_error"}
        mask = np.ones(len(yt), bool)
        for s in scores.values():
            mask &= np.isfinite(s)
        yc = yt[mask]
        row.update(n_eval=int(mask.sum()))
        if mask.sum() < MIN_TEST_ROWS:
            return {**row, "status": "too_short"}
        if yc.sum() == 0 or yc.sum() == len(yc):
            return {**row, "status": "no_anomaly_in_test"}
        runs = _runs(yc)
        lens = np.array([e - s + 1 for s, e in runs])
        mb = int(np.clip(np.median(lens), 2, max(2, len(yc) // 10)))
        row.update(prevalence=float(yc.mean()), n_events=len(runs), median_len=float(np.median(lens)),
                   max_len=int(lens.max()), mean_len=float(lens.mean()))
        for d, s in scores.items():
            row[f"auc_pr_{d}"] = auc_pr(yc, s[mask])
            row[f"vus_pr_{d}"] = vus_pr(yc, s[mask], max_buffer=mb, n_buffers=5)
    except Exception as e:                                           # noqa: BLE001
        return {**row, "status": f"error: {type(e).__name__}: {str(e)[:100]}"}
    row["seconds"] = round(time.time() - t0, 2)
    return row


# ------------------------------------------------------------------ run
def cmd_run(a):
    files = sorted(glob.glob(os.path.join(a.data, "**", "*.csv"), recursive=True))
    if a.datasets:
        files = [f for f in files if parse_name(f)[0] in set(a.datasets)]
    if a.max_series:
        files = files[: a.max_series]
    if not files:
        sys.exit(f"no CSV files under {a.data}. Download TSB-AD-U.zip from thedatum.org and unzip it there.")
    x, y = load_series(files[0])
    print(f"schema check on {os.path.basename(files[0])}: n={len(x)}, anomalies={int(y.sum())}, "
          f"parsed (dataset, train_len)={parse_name(files[0])}")
    done = set(pd.read_csv(a.out).file) if os.path.exists(a.out) else set()
    todo = [f for f in files if os.path.basename(f) not in done]
    print(f"{len(files)} files, {len(done)} already done, {len(todo)} to run, {a.jobs} job(s)")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)

    # rows can have different columns (err_* appear only on failures), so rewrite the CSV per chunk
    rows = []
    def flush():
        if not rows:
            return
        new = pd.DataFrame(rows)
        old = pd.read_csv(a.out) if os.path.exists(a.out) else pd.DataFrame()
        pd.concat([old, new], ignore_index=True).to_csv(a.out, index=False)
        rows.clear()

    t0 = time.time()
    if a.jobs > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=a.jobs) as ex:
            futs = [ex.submit(evaluate_file, f, a.train_frac, a.seed) for f in todo]
            for i, fu in enumerate(as_completed(futs), 1):
                rows.append(fu.result())
                if i % a.chunk == 0:
                    flush(); print(f"  {i}/{len(todo)}  {time.time() - t0:.0f}s")
    else:
        for i, f in enumerate(todo, 1):
            rows.append(evaluate_file(f, a.train_frac, a.seed))
            if i % a.chunk == 0:
                flush(); print(f"  {i}/{len(todo)}  {time.time() - t0:.0f}s")
    flush()
    r = pd.read_csv(a.out)
    print("status counts:\n", r.status.value_counts().to_string())


# ------------------------------------------------------------------ analyse
def md(df):
    cols = list(df.columns)
    return "\n".join(["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)] +
                     ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)])


def cmd_analyse(a):
    raw = pd.read_csv(a.results)
    ok = raw[raw.status == "ok"].copy()
    dets = [d for d in DETECTORS if f"auc_pr_{d}" in ok.columns]
    ok["log_len"] = np.log(ok.median_len)
    ok["log_prev"] = np.log(ok.prevalence)
    out = ["# Step 8 report (generated by experiments/public_benchmark.py analyse)", "",
           "Pre-registration: `docs/STEP8_PREREGISTRATION.md`. Detectors are CPU stand-ins; VUS-PR is our re-implementation.", "",
           "## Series accounting", "", md(raw.status.value_counts().rename_axis("status").reset_index(name="n_series")), "",
           f"Analysed: {len(ok)} series from {ok.dataset.nunique()} source datasets.", ""]
    errs = {d: int(raw[f"err_{d}"].notna().sum()) for d in DETECTORS if f"err_{d}" in raw.columns}
    if errs:
        out += ["Detector errors (series on which a detector raised): " + ", ".join(f"{k}: {v}" for k, v in errs.items()), ""]

    A, B = PRIMARY
    for metric, tag in (("auc_pr", "PRIMARY (pre-registered)"), ("vus_pr", "exploratory")):
        ok["d"] = ok[f"{metric}_{A}"] - ok[f"{metric}_{B}"]
        r, lo, hi, per = stratified_partial_spearman(ok, "log_len", "d", "log_prev", n_boot=a.n_boot, seed=a.seed)
        if np.isfinite(r) and lo > 0:
            verdict = "SUPPORTED (reconstruction gains on longer events; interval excludes 0)"
        elif np.isfinite(r) and hi < 0:
            verdict = "CONTRADICTED (density gains on longer events; interval excludes 0)"
        else:
            verdict = "NOT SUPPORTED (interval contains 0 or too few datasets)"
        out += [f"## {tag}: {A} minus {B}, {metric}",
                f"Partial Spearman between log median event length and the metric difference, controlling for log prevalence, "
                f"stratified by source dataset ({len(per)} datasets with >= 8 series), Fisher-z combined, bootstrap over datasets.", "",
                f"**rho = {r:.3f}, 95% interval [{lo:.3f}, {hi:.3f}] -> {verdict}**", "",
                md(per.round(3).rename(columns={"group": "dataset"})) if len(per) else "(no dataset with enough series)", ""]

    out += ["## Exploratory: other pairs (AUC-PR; no multiplicity correction)", ""]
    rows = []
    for A2, B2 in (("window_pca", "isolation_forest"), ("matrix_profile", "window_pca"), ("seasonal_naive", "window_pca"),
                   ("matrix_profile", "knn_inductive"), ("window_iforest", "window_pca")):
        if f"auc_pr_{A2}" in ok.columns and f"auc_pr_{B2}" in ok.columns:
            ok["d"] = ok[f"auc_pr_{A2}"] - ok[f"auc_pr_{B2}"]
            r, lo, hi, per = stratified_partial_spearman(ok, "log_len", "d", "log_prev", n_boot=a.n_boot, seed=a.seed)
            rows.append(dict(pair=f"{A2} - {B2}", rho=round(r, 3), lo=round(lo, 3), hi=round(hi, 3), n_datasets=len(per)))
    out += [md(pd.DataFrame(rows)) if rows else "(none)", ""]

    out += ["## Mean within-series rank by event-length tercile (1 = best; AUC-PR)", ""]
    ok["len_bin"] = pd.qcut(ok.median_len, 3, duplicates="drop")
    bins = ok.len_bin.cat.categories if hasattr(ok.len_bin, "cat") else None
    if bins is not None and len(bins) >= 2:
        edges = [bins[0].left] + [b.right for b in bins]
        t = mean_rank_by_bin(ok, dets, "auc_pr_{}", "median_len", edges).round(2)
        t.insert(0, "n_series", ok.groupby(pd.cut(ok.median_len, edges, include_lowest=True), observed=True).size().values)
        out += [md(t.reset_index().rename(columns={"median_len": "median event length (samples)"})), ""]
    ov = pd.DataFrame({"mean_auc_pr": [ok[f"auc_pr_{d}"].mean() for d in dets],
                       "mean_lift": [(ok[f"auc_pr_{d}"] / ok.prevalence).mean() for d in dets],
                       "mean_vus_pr": [ok[f"vus_pr_{d}"].mean() for d in dets]}, index=dets).round(3)
    out += ["## Overall means (all analysed series)", "", md(ov.reset_index().rename(columns={"index": "detector"})), ""]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out, "w").write("\n".join(out))
    print("\n".join(out[:12]))
    print("wrote", a.out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--data", required=True); r.add_argument("--out", required=True)
    r.add_argument("--jobs", type=int, default=1); r.add_argument("--max-series", type=int, default=0)
    r.add_argument("--datasets", nargs="*"); r.add_argument("--train-frac", type=float, default=0.1)
    r.add_argument("--seed", type=int, default=0); r.add_argument("--chunk", type=int, default=20)
    n = sub.add_parser("analyse")
    n.add_argument("--results", required=True); n.add_argument("--out", default="docs/step8_report.md")
    n.add_argument("--n-boot", type=int, default=500); n.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cmd_run(args) if args.cmd == "run" else cmd_analyse(args)
