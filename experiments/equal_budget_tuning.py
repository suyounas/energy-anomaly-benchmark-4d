"""Equal-budget tuning + evaluation on identical rows (register rows 10, 17; W2; R3-3).

Protocol
  1. Take the clean training window (e.g. GB Protocol B: 2018-06-01..2020-02-01).
  2. Chronologically split it 70/30 into FIT and VALIDATION; inject SYNTHETIC events into the
     validation part only.  The real test events are never used for any choice.
  3. For each detector family: evaluate the default config + N random configs (same N for every
     family), select by validation AUC-PR.
  4. Refit on the full training window with the selected config (and with the default config), for
     several seeds; score the real test set.  ALL detectors are evaluated on the SAME rows (the
     intersection of rows every model can score) at the SAME 5% alert budget, and also with
     budget-free AUC-PR.
Usage (classical only, runs anywhere):
  python experiments/equal_budget_tuning.py --csv neso_demand_full.csv --time-col Datetime \
      --value-col ND_MW --base-minutes 30 --factor 2 --event 2020-03-23 2020-05-31 COVID \
      --event 2022-12-08 2022-12-17 ColdSnap --train-window 2018-06-01 2020-02-01 \
      --families isolation_forest knn_inductive rolling_z --out results/real/gb_equal_budget
Add  lstm_ae transformer  on a GPU/Colab runtime (needs torch).
"""
import argparse, os, sys
import numpy as np, pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ebench import timestamp_level_prf, event_level_prf, per_event_coverage, auc_pr, event_mask
from ebench.synthetic import resample
from ebench.detectors import top_pct_flag
from ebench.tuning import FAMILIES, equal_budget_search, inject_into_series
from real_resolution_ablation import build_features, FEATS


def main(a):
    df = pd.read_csv(a.csv, parse_dates=[a.time_col]).sort_values(a.time_col)
    t0 = pd.DatetimeIndex(df[a.time_col]); x0 = df[a.value_col].to_numpy(float)
    t, x, _ = resample(t0, x0, np.zeros(len(x0), int), a.factor)
    spd = int(round(24 * 60 / (a.base_minutes * a.factor)))
    events = [(s, e, n) for s, e, n in a.event]
    lo, hi = pd.Timestamp(a.train_window[0]), pd.Timestamp(a.train_window[1])
    tr_mask = (t >= lo) & (t < hi)
    # --- validation with synthetic events: inject on the NATIVE grid inside the clean training
    #     window (last 30% only), then resample the window, so durations stay in days at any factor
    m0 = (t0 >= lo) & (t0 < hi)
    n_fit_native = int(0.7 * m0.sum()); n_fit_native -= n_fit_native % 48
    x_inj_nat, y_inj_nat = inject_into_series(t0[m0], x0[m0], n_fit_native, a.val_durations,
                                              a.val_prevalence, a.val_magnitude, a.seed,
                                              base_minutes=a.base_minutes)
    t_w, x_inj, y_inj = resample(t0[m0], x_inj_nat, y_inj_nat, a.factor)
    F_inj = build_features(t_w, x_inj, spd, a.window_days)
    y_all = pd.Series(y_inj, index=t_w).reindex(F_inj.index).fillna(0).to_numpy().astype(int)
    split_t = t_w[min(n_fit_native // a.factor, len(t_w) - 1)]
    Ffit, Fval = F_inj[F_inj.index < split_t], F_inj[F_inj.index >= split_t]
    yval = y_all[F_inj.index >= split_t]
    print(f"fit rows {len(Ffit)}, validation rows {len(Fval)}, validation prevalence {yval.mean():.3f}")

    trials, chosen = equal_budget_search(a.families, Ffit, Fval, yval, FEATS,
                                         n_trials=a.n_trials, seed=a.seed, spd=spd)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    trials.to_csv(a.out + "_trials.csv", index=False)
    print(trials.groupby("family").agg(val_min=("val_auc_pr", "min"), val_max=("val_auc_pr", "max"),
                                       val_lift_max=("val_lift", "max"), informative=("informative", "first")).round(3))
    for fam_name, ok in trials.groupby("family").informative.first().items():
        if not ok:
            print(f"NOTE {fam_name}: validation cannot separate its configs (spread < 0.02); "
                  f"the default config is used, 'tuned' == 'default' for this family")

    # --- final evaluation on real data, identical rows for every detector
    F = build_features(t, x, spd, a.window_days)
    tt = F.index
    in_tr = (tt >= lo) & (tt < hi)
    Ftr, Fte = F[in_tr], F[~in_tr]
    y = np.zeros(len(F), int)
    for s, e, _ in events:
        y[event_mask(tt, s, e)] = 1
    yte, tte = y[~in_tr], tt[~in_tr]
    score_store = {}
    for name in a.families:
        for tag, cfg in (("default", FAMILIES[name]["default"]), ("tuned", chosen[name])):
            for seed in range(a.n_seeds):
                score_store[(name, tag, seed)] = FAMILIES[name]["fn"](cfg, Ftr, Fte, FEATS, seed, spd=spd)
    common = np.ones(len(Fte), bool)
    for sc in score_store.values():
        common &= ~np.isnan(sc)
    print(f"common evaluation rows: {common.sum()} / {len(Fte)}")
    yc, tc = yte[common], tte[common]
    rows = []
    for (name, tag, seed), sc in score_store.items():
        s = sc[common]; flag = top_pct_flag(s, 95)
        tsm = timestamp_level_prf(yc, flag)
        evm = event_level_prf(tc, flag, events, tol=pd.Timedelta(days=a.tol_days))
        r = dict(family=name, config=tag, seed=seed, n_eval=int(common.sum()), prevalence=float(yc.mean()),
                 auc_pr=auc_pr(yc, s), ts_precision=tsm["precision"], ts_recall=tsm["recall"], ts_f1=tsm["f1"],
                 ev_precision=evm["precision"], ev_recall=evm["recall"], ev_f1=evm["f1"], n_events=evm["n_events"],
                 cfg=str(chosen[name] if tag == "tuned" else FAMILIES[name]["default"]))
        for en, c in per_event_coverage(tc, flag, events).items():
            r[f"cov_{en}"] = c["coverage"]
        rows.append(r)
    res = pd.DataFrame(rows)
    res.to_csv(a.out + "_final.csv", index=False)
    agg = res.groupby(["family", "config"])[["auc_pr", "ts_f1", "ev_f1"] + [c for c in res if c.startswith("cov_")]].agg(["mean", "std"]).round(3)
    pd.set_option("display.width", 250); print(agg)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True); ap.add_argument("--time-col", required=True)
    ap.add_argument("--value-col", required=True); ap.add_argument("--base-minutes", type=int, required=True)
    ap.add_argument("--factor", type=int, default=1)
    ap.add_argument("--event", nargs=3, action="append", metavar=("START", "END", "NAME"), required=True)
    ap.add_argument("--train-window", nargs=2, required=True)
    ap.add_argument("--families", nargs="+", default=["isolation_forest", "knn_inductive", "rolling_z"])
    ap.add_argument("--n-trials", type=int, default=8); ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--window-days", type=float, default=30.0)
    ap.add_argument("--val-durations", type=int, nargs="+", default=[1, 3])
    ap.add_argument("--val-prevalence", type=float, default=0.08)
    ap.add_argument("--val-magnitude", type=float, default=3.0)
    ap.add_argument("--tol-days", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/real/equal_budget")
    main(ap.parse_args())
