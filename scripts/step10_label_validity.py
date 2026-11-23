#!/usr/bin/env python3
"""Step 10 label validity (pre-registered Analyses A/B). Uses only the demand series and the labels, no detector scores.
Per series: causal weekly seasonal-naive residual r_t = x_t - x_{t-168h}; z_t = (r_t - median) / (1.4826 MAD) with median and MAD over
the whole series; daily |z| = mean of |z_t| over the calendar day. Per event: mean daily |z| inside the event window, versus the
distribution of daily |z| on days outside every event of the same series. An event is 'validated' if its mean daily |z| exceeds the
95th percentile of daily |z| outside events (secondary criterion; series and labels only)."""
import argparse, os, sys
import numpy as np, pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument("--eia", default="data/step10/eia930")
    ap.add_argument("--events-final", default="results/step10/runs_classical/events_final.csv")
    ap.add_argument("--manifest", default="results/step10/series_manifest.csv")
    ap.add_argument("--out", default="results/step10/label_validity")
    ap.add_argument("--q", type=float, default=0.95)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    sys.path.insert(0, os.path.join(a.repo, "scripts")); sys.path.insert(0, a.repo); sys.path.insert(0, os.path.join(a.repo, "experiments"))
    from step10_run import load_series
    ev = pd.read_csv(a.events_final, parse_dates=["start", "end"])
    man = pd.read_csv(a.manifest); bas = sorted(set(man[man.qualifies].ba) & set(ev.ba))
    out, base = [], []
    for ba in bas:
        x = load_series(a.eia, ba).astype(float)
        r = x - x.shift(168, freq="h").reindex(x.index)
        r = r.dropna(); med = r.median(); mad = 1.4826 * (r - med).abs().median()
        z = ((r - med) / mad).abs()
        daily = z.resample("D").mean().dropna()
        e = ev[ev.ba == ba]
        inside = pd.Series(False, index=daily.index)
        for s, t in zip(e.start, e.end): inside |= (daily.index >= s.normalize()) & (daily.index <= t.normalize())
        outside = daily[~inside]
        thr = outside.quantile(a.q)
        base.append(dict(ba=ba, n_days_outside=len(outside), outside_median=outside.median(), outside_p95=thr))
        for row in e.itertuples():
            d = daily[(daily.index >= row.start.normalize()) & (daily.index <= row.end.normalize())]
            if d.empty: continue
            m = d.mean()
            out.append(dict(ba=ba, event_id=row.event_id, storm_group=row.storm_group, source_class=row.source_class, family=row.family,
                            mean_daily_absz=m, outside_p95=thr, ratio_to_outside_median=m / outside.median(), validated=bool(m > thr),
                            pct_rank_vs_outside=float((outside < m).mean())))
    ev_out = pd.DataFrame(out); ev_out.to_csv(f"{a.out}/event_validity.csv", index=False)
    pd.DataFrame(base).to_csv(f"{a.out}/series_baselines.csv", index=False)
    summ = []
    for name, g in [("all", ev_out), *[(f"class_{c}", g) for c, g in ev_out.groupby("source_class")],
                    *[(f"family_{f}", ev_out[ev_out.family.str.contains(f)]) for f in ["heat", "cold", "tropical"]]]:
        summ.append(dict(view=name, n_events=len(g), share_validated=g.validated.mean(), median_pct_rank=g.pct_rank_vs_outside.median(),
                         median_ratio=g.ratio_to_outside_median.median(), n_storm_groups=g.storm_group.nunique()))
    s = pd.DataFrame(summ); s.to_csv(f"{a.out}/summary.csv", index=False); print(s.round(3).to_string(index=False))

if __name__ == "__main__":
    main()
