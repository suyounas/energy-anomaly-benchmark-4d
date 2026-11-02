"""Step 9: factor attribution on a fully crossed semi-synthetic design (pre-registered in docs/STEP9_PREREGISTRATION.md).

  python3 experiments/factor_attribution.py run --out results/factorial/grid.csv --jobs 4          # ~10-30 min
  python3 experiments/factor_attribution.py analyse --results results/factorial/grid.csv --out docs/step9_report.md

Each dataset (background, event type, duration, magnitude, resolution, replicate) is generated once, features are built once,
and every detector is scored under both feature sets.  Detectors that ignore the feature columns give identical scores under
both (structural zero).  All detectors are scored on the SAME rows.  CPU only.
"""
import argparse, itertools, os, sys, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
from ebench import auc_pr
from ebench.synthetic import resample
from ebench.tuning import FAMILIES
from ebench.factorial import (BACKGROUNDS, EVENT_TYPES, make_background, residual_scale, inject_events,
                              ss_decomposition, ranking_stability)
from real_resolution_ablation import build_features, feature_cols

DETECTORS = ["diff_z", "rolling_z", "seasonal_naive", "isolation_forest", "knn_inductive", "window_pca",
             "window_iforest", "matrix_profile", "spectral_residual"]
USES_FEATURES = {"isolation_forest", "knn_inductive", "window_pca", "window_iforest"}
LEVELS = dict(background=list(BACKGROUNDS), event=list(EVENT_TYPES), dur_h=[4, 24, 120], mag=[3, 6, 12], res_h=[1, 4])
FACTORS = ["background", "event", "dur_h", "mag", "res_h", "features", "detector"]
TRAIN_DAYS, TEST_DAYS = 120, 90


def run_dataset(spec, detectors=DETECTORS, train_days=TRAIN_DAYS, test_days=TEST_DAYS, n_events=3):
    bg, ev, dur, mag, res, rep = (spec[k] for k in ("background", "event", "dur_h", "mag", "res_h", "rep"))
    base = dict(background=bg, event=ev, dur_h=dur, mag=mag, res_h=res, rep=rep)
    rng = np.random.default_rng([LEVELS["background"].index(bg), LEVELS["event"].index(ev), dur, mag, res, rep])
    n_h = 24 * (train_days + test_days)
    t, x = make_background(bg, n_h, rng)
    test_start = 24 * train_days
    scale = residual_scale(x[:test_start])
    x2, y = inject_events(x, test_start, ev, dur, mag, scale, rng, n_events=n_events)
    t2, xr, yr = resample(t, x2, y, res)
    spd = 24 // res
    F = build_features(t2, xr, spd, window_days=5, with_resid=True)
    yr = pd.Series(yr, index=t2).loc[F.index].to_numpy()
    cut = t2[test_start // res]
    tr, te = F.index < cut, F.index >= cut
    Ftr, Fev, yev = F[tr], F[te], yr[te]
    if yev.sum() == 0:
        return []
    scores = {}
    for f in ("recipe", "resid"):
        cols = feature_cols(f)
        for d in detectors:
            if f == "resid" and d not in USES_FEATURES:
                scores[(f, d)] = scores[("recipe", d)]
                continue
            fam = FAMILIES[d]
            c = cols if d in USES_FEATURES else feature_cols("resid")
            try:
                scores[(f, d)] = np.asarray(fam["fn"](dict(fam["default"]), Ftr, Fev, c, rep, spd=spd), float)
            except Exception as e:                                        # noqa: BLE001  (counted by the caller)
                scores[(f, d)] = np.full(len(Fev), np.nan)
    mask = np.ones(len(yev), bool)
    for s in scores.values():
        mask &= np.isfinite(s)
    ym = yev[mask]
    if ym.sum() == 0 or ym.sum() == len(ym) or mask.sum() < 100:
        return []
    prev = ym.mean()
    rows = []
    for (f, d), s in scores.items():
        ap = auc_pr(ym, s[mask])
        rows.append(dict(**base, features=f, detector=d, prevalence=prev, auc_pr=ap, skill=(ap - prev) / (1 - prev), n_eval=int(mask.sum())))
    return rows


def _job(spec):
    try:
        return spec, run_dataset(spec), None
    except Exception as e:                                                # noqa: BLE001
        return spec, [], f"{type(e).__name__}: {str(e)[:100]}"


def cmd_run(a):
    lv = {k: (v[:2] if a.quick else v) for k, v in LEVELS.items()}          # --quick: 2 levels per factor (still balanced)
    specs = [dict(zip(["background", "event", "dur_h", "mag", "res_h", "rep"], c))
             for c in itertools.product(lv["background"], lv["event"], lv["dur_h"], lv["mag"], lv["res_h"], range(a.reps))]
    if a.max_datasets:
        specs = specs[: a.max_datasets]
    done = set()
    if os.path.exists(a.out):
        old = pd.read_csv(a.out)
        done = set(map(tuple, old[["background", "event", "dur_h", "mag", "res_h", "rep"]].drop_duplicates().itertuples(index=False)))
    todo = [s for s in specs if tuple(s[k] for k in ("background", "event", "dur_h", "mag", "res_h", "rep")) not in done]
    print(f"{len(specs)} datasets, {len(done)} done, {len(todo)} to run, {a.jobs} job(s)")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    buf, fails, t0 = [], [], time.time()

    def flush():
        if buf:
            new = pd.DataFrame(buf)
            old = pd.read_csv(a.out) if os.path.exists(a.out) else pd.DataFrame()
            pd.concat([old, new], ignore_index=True).to_csv(a.out, index=False)
            buf.clear()

    def take(res, i):
        spec, rows, err = res
        if err or not rows:
            fails.append((spec, err or "no usable rows"))
        buf.extend(rows)
        if i % a.chunk == 0:
            flush(); print(f"  {i}/{len(todo)}  {time.time() - t0:.0f}s")

    if a.jobs > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=a.jobs) as ex:
            for i, fu in enumerate(as_completed([ex.submit(_job, s) for s in todo]), 1):
                take(fu.result(), i)
    else:
        for i, s in enumerate(todo, 1):
            take(_job(s), i)
    flush()
    print(f"datasets with no usable rows or errors: {len(fails)}")
    for spec, err in fails[:10]:
        print("  ", spec, err)


def md(df):
    cols = list(df.columns)
    return "\n".join(["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)] +
                     ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)])


def complete_design(df):
    """Keep only datasets that produced every (features, detector) row, then only reps present everywhere."""
    key = ["background", "event", "dur_h", "mag", "res_h", "rep"]
    n_per = df.groupby(key).size()
    full = n_per[n_per == n_per.max()].index
    d = df.set_index(key).loc[full].reset_index()
    cells = ["background", "event", "dur_h", "mag", "res_h"]
    r = int(d.groupby(cells).rep.nunique().min())
    # use the first r replicates of every cell so the design is balanced
    first = {k: sorted(g.rep.unique())[:r] for k, g in d.groupby(cells)}
    keep = d[[rep in first[tuple(row)] for row, rep in zip(d[cells].itertuples(index=False, name=None), d.rep)]]
    keep = keep.copy()
    keep["rep"] = keep.groupby(cells).rep.transform(lambda x: x.rank(method="dense").astype(int) - 1)   # relabel 0..r-1 per cell
    return keep, len(n_per) - len(full), r


def cmd_analyse(a):
    raw = pd.read_csv(a.results)
    df, n_incomplete, r = complete_design(raw)
    n_ds = df[["background", "event", "dur_h", "mag", "res_h", "rep"]].drop_duplicates().shape[0]
    out = ["# Step 9 report (generated by experiments/factor_attribution.py analyse)", "",
           "Pre-registration: `docs/STEP9_PREREGISTRATION.md`. Outcome = normalised AUC-PR (skill). Synthetic energy-like data; default detector configs.", "",
           f"Datasets in the balanced design: {n_ds} ({r} replicates per cell); datasets dropped as incomplete: {n_incomplete}.", ""]
    ss = ss_decomposition(df, FACTORS, "skill", max_order=2)
    rng = np.random.default_rng(a.seed)
    cells = ["background", "event", "dur_h", "mag", "res_h"]
    reps = sorted(df.rep.unique())
    boots = []
    for _ in range(a.n_boot):
        pick = rng.choice(reps, size=len(reps), replace=True)
        parts = [df[df.rep == rp].assign(rep=j) for j, rp in enumerate(pick)]
        b = pd.concat(parts, ignore_index=True)
        boots.append(ss_decomposition(b, FACTORS, "skill", max_order=2).set_index("term").share)
    B = pd.concat(boots, axis=1)
    ss["lo"] = ss.term.map(B.quantile(0.025, axis=1)); ss["hi"] = ss.term.map(B.quantile(0.975, axis=1))
    show = ss.sort_values("share", ascending=False).copy()
    for c in ("share", "lo", "hi"):
        show[c] = (100 * show[c]).round(1)
    out += ["## Share of total variance in skill (percent; interval = bootstrap over replicates only)", "",
            md(show[["term", "share", "lo", "hi"]].head(20)), ""]
    det_main = float(ss.loc[ss.term == "detector", "share"].iloc[0])
    det_all = float(ss.loc[ss.term.str.contains("detector"), "share"].sum())
    out += [f"**Detector main effect: {100 * det_main:.1f}%. Everything that involves the detector (main + interactions): {100 * det_all:.1f}%.** "
            f"Share of detector-related variance that is conditional on the setting: {100 * (1 - det_main / det_all):.0f}%.", ""]
    summ, per = ranking_stability(df, cells + ["features"], "detector", "skill")
    out += ["## Ranking stability (conditions = background x event x duration x magnitude x resolution x features, mean over replicates)", "",
            f"* Global best detector: **{summ['global_best']}**; it is best in {100 * summ['share_conditions_global_best_wins']:.0f}% of {summ['n_conditions']} conditions.",
            f"* Mean Spearman correlation between a condition's detector ranking and the global ranking: {summ['mean_spearman']:.2f}.",
            f"* Distinct winners across conditions: {summ['n_distinct_winners']} of {len(DETECTORS)} detectors.", ""]
    win = per.groupby(["event"]).winner.value_counts(normalize=True).rename("share").reset_index()
    wt = win.pivot(index="event", columns="winner", values="share").fillna(0).round(2)
    out += ["Share of conditions won by each detector, by event type:", "", md(wt.reset_index()), ""]
    wf = per.groupby("features").winner.value_counts(normalize=True).rename("share").reset_index()
    out += ["By feature set:", "", md(wf.pivot(index="features", columns="winner", values="share").fillna(0).round(2).reset_index()), ""]
    mean = df.groupby("detector").skill.mean().sort_values(ascending=False).round(3)
    out += ["## Mean skill by detector (all conditions)", "", md(mean.rename("mean_skill").reset_index()), ""]
    for f in ("event", "dur_h", "mag", "res_h", "features", "background"):
        t = df.groupby(["detector", f]).skill.mean().unstack(f).round(3).reset_index()
        out += [f"## Mean skill: detector x {f}", "", md(t), ""]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out, "w").write("\n".join(out))
    print("\n".join(out[:28]))
    print("wrote", a.out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", required=True); r.add_argument("--jobs", type=int, default=1)
    r.add_argument("--reps", type=int, default=4); r.add_argument("--max-datasets", type=int, default=0)
    r.add_argument("--chunk", type=int, default=20)
    r.add_argument("--quick", action="store_true", help="2 levels per factor (tests / smoke runs); not the pre-registered design")
    n = sub.add_parser("analyse")
    n.add_argument("--results", required=True); n.add_argument("--out", default="docs/step9_report.md")
    n.add_argument("--n-boot", type=int, default=200); n.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cmd_run(args) if args.cmd == "run" else cmd_analyse(args)
