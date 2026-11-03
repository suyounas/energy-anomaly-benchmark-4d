"""Annotation-validity analysis.  Detector scores are computed once (default configs, inductive, same rows for
every detector); only the LABELS change.  Output answers three questions without relying on detectors "finding unlabelled events":

  A. Does the series deviate from its causal seasonal expectation inside each labelled event?   (event_effects)
  B. How many days outside the labelled events deviate at least as much as the weakest labelled event?  (candidates, not claimed anomalies)
  C. If boundaries are off by days, an event is dropped, or events are longer/shorter, do detector rankings change?

  python3 experiments/label_sensitivity.py --csv neso_demand_full.csv --time-col Datetime --value-col ND_MW --base-minutes 30 --factor 2 \
      --event 2020-03-23 2020-05-31 COVID --event 2022-12-08 2022-12-17 ColdSnap --train-window 2018-06-01 2020-02-01 \
      --features resid --families isolation_forest knn_inductive rolling_z diff_z seasonal_naive window_pca matrix_profile --out results/label_sens/gb
"""
import argparse, os, sys, warnings
import numpy as np, pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ebench.synthetic import resample
from ebench.tuning import FAMILIES
from ebench.labelsens import evaluate_scenarios, rank_robustness, event_effects, unlabelled_days
from real_resolution_ablation import build_features, feature_cols, seasonal_residual


def md(df):
    cols = list(df.columns)
    return "\n".join(["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)] +
                     ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)])


def main(a):
    df = pd.read_csv(a.csv, parse_dates=[a.time_col]).sort_values(a.time_col)
    t0 = pd.DatetimeIndex(df[a.time_col]); x0 = df[a.value_col].to_numpy(float)
    t, x, _ = resample(t0, x0, np.zeros(len(x0), int), a.factor)
    spd = int(round(24 * 60 / (a.base_minutes * a.factor)))
    events = [(s, e, n) for s, e, n in a.event]
    lo, hi = pd.Timestamp(a.train_window[0]), pd.Timestamp(a.train_window[1])
    F = build_features(t, x, spd, a.window_days, with_resid=True)
    in_tr = (F.index >= lo) & (F.index < hi)
    Ftr, Fte = F[in_tr], F[~in_tr]
    cols = feature_cols(a.features)
    scores = {}
    for name in a.families:
        fam = FAMILIES[name]
        if any(c not in F.columns for c in fam.get("requires", [])):
            print(f"SKIP {name}"); continue
        use = cols if name not in ("seasonal_naive",) else feature_cols("resid")
        per_seed = [np.asarray(fam["fn"](dict(fam["default"]), Ftr, Fte, use, s, spd=spd), float) for s in range(a.n_seeds)]
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)      # rows where every seed is NaN (window warm-up) are dropped below
            scores[name] = np.nanmean(per_seed, axis=0)
    common = np.ones(len(Fte), bool)
    for s in scores.values():
        common &= np.isfinite(s)
    tc = Fte.index[common]
    scores = {k: v[common] for k, v in scores.items()}
    out = ["# Label sensitivity report (experiments/label_sensitivity.py)", "",
           f"Series: {a.csv}; test rows {common.sum()}; detectors (default configs, mean score over {a.n_seeds} seeds): {', '.join(scores)}.", ""]

    # A and B: detector-independent evidence about the labels
    resid = seasonal_residual(pd.Series(x, index=t), spd)
    rs = pd.Series(resid, index=t)
    ee = event_effects(rs.index, rs.to_numpy(), events)
    out += ["## A. Does the series deviate inside each labelled event? (causal seasonal-naive residual, robust z)", "",
            "`mean_daily_dev` = mean over the event's days of the daily mean |z|; `pct_of_ref_days_below` = percentage of non-event days with a smaller daily deviation.", "",
            md(ee.round(3)), ""]
    n_c, n_out, cand, thr = unlabelled_days(rs.index, rs.to_numpy(), events)
    out += ["## B. Unlabelled days that deviate at least as much as the weakest labelled event",
            f"{n_c} of {n_out} days outside every labelled event reach the weakest event's daily deviation ({thr:.2f}). These are candidates for missing labels, "
            "**not** claimed anomalies (a day can deviate for reasons such as holidays or weather that nobody labelled as an event).", ""]
    if n_c:
        cand["month"] = pd.to_datetime(cand["day"]).dt.to_period("M").astype(str)
        out += ["Candidate days by month (top 12):", "", md(cand.groupby("month").size().rename("days").reset_index().sort_values("days", ascending=False).head(12)), ""]
    cand.to_csv(a.out + "_candidate_days.csv", index=False)

    # C: label perturbations
    sc_df = evaluate_scenarios(tc, scores, events)
    sc_df.to_csv(a.out + "_scenarios.csv", index=False)
    out += ["## C. Detector ranking under label perturbations", ""]
    for metric in ("auc_pr", "ts_f1"):
        summ, per = rank_robustness(sc_df, metric)
        out += [f"### {metric}",
                f"Baseline best: **{summ['baseline_best']}**; unchanged in {100 * summ['share_best_unchanged']:.0f}% of {summ['n_scenarios']} perturbed label sets; "
                f"Kendall tau of the full ranking vs baseline: median {summ['median_tau']:.2f}, minimum {summ['min_tau']:.2f}.", "",
                md(per.round(3)), ""]
        w = sc_df.pivot(index="scenario", columns="detector", values=metric).round(3)
        out += [md(w.reset_index()), ""]
    open(a.out + "_report.md", "w").write("\n".join(out))
    print("\n".join(out[:30])); print("wrote", a.out + "_report.md")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True); ap.add_argument("--time-col", required=True)
    ap.add_argument("--value-col", required=True); ap.add_argument("--base-minutes", type=int, required=True)
    ap.add_argument("--factor", type=int, default=1)
    ap.add_argument("--event", nargs=3, action="append", metavar=("START", "END", "NAME"), required=True)
    ap.add_argument("--train-window", nargs=2, required=True)
    ap.add_argument("--families", nargs="+", default=["isolation_forest", "knn_inductive", "rolling_z", "diff_z", "seasonal_naive", "window_pca", "matrix_profile"])
    ap.add_argument("--features", choices=["recipe", "resid"], default="resid")
    ap.add_argument("--n-seeds", type=int, default=3); ap.add_argument("--window-days", type=float, default=30.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    main(a)
