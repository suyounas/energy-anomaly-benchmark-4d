"""ERCOT label-tier diagnostic (register rows 5 and 9; reviewer R3-4).

Replaces the earlier diagnostic that reported "94% of the LSTM's flags fall in summer 2023"
(1,639 of 1,751) from a script that was never committed.  This one is in the repository,
tested, and works from the per-timestamp flags that `code/study4_ercot_eia/02_full_pipeline.py`
now writes to `ercot_flags.csv` (long format: protocol, detector, time, flag).

Label tiers
  Tier 1  events fixed before any modelling (Texas Freeze, Elliott spike).
  Tier 2  ERCOT's documented all-time demand records, 2023-06-27 .. 2023-08-10.  This window was
          noticed AFTER seeing detector output, so it is a post-hoc sensitivity analysis only.

For every (protocol, detector) the report gives
  * where the flags go: share inside Tier 1, inside Tier 2, in Jul-Sep 2023, and outside every
    label (the quantity the old "94%" figure tried to describe);
  * timestamp-level precision / recall / F1 under Tier 1 alone and under Tier 1 + Tier 2, each with
    a day-block bootstrap interval;
  * per-event coverage for every Tier 1 event and for the Tier 2 window.
A window outside a protocol's test span is reported as `in_test_span = False`, never as zero.

Usage:
  python experiments/ercot_label_tiers.py --flags ercot_flags.csv --out results/real/ercot_tiers.csv
"""
import argparse, os, sys
import numpy as np, pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ebench import (event_mask, per_event_coverage, timestamp_level_prf, block_bootstrap_ci)

TIER1 = [("2021-02-14", "2021-02-19", "Texas Freeze"), ("2022-12-22", "2022-12-26", "Elliott spike")]
TIER2 = [("2023-06-27", "2023-08-10", "2023 record demand")]
SUMMER_2023 = ("2023-07-01", "2023-09-30")


def _mask_any(times, windows):
    m = np.zeros(len(times), bool)
    for s, e, *_ in windows:
        m |= event_mask(times, s, e)
    return m


def _f1_on(y, flag, idx):
    return timestamp_level_prf(y[idx], flag[idx])["f1"]


def tier_report(flags, tier1=TIER1, tier2=TIER2, n_boot=500, seed=0):
    """`flags` is a long DataFrame with columns protocol, detector, time, flag.
    Returns a DataFrame with one row per (protocol, detector)."""
    need = {"protocol", "detector", "time", "flag"}
    if not need <= set(flags.columns):
        raise ValueError(f"flags file needs columns {sorted(need)}, got {list(flags.columns)}")
    flags = flags.copy()
    flags["time"] = pd.to_datetime(flags["time"])
    rows = []
    for (proto, det), g in flags.groupby(["protocol", "detector"], sort=False):
        g = g.sort_values("time")
        t = pd.DatetimeIndex(g["time"]); f = g["flag"].to_numpy().astype(int)
        n_flags = int(f.sum())
        m1, m2 = _mask_any(t, tier1), _mask_any(t, tier2)
        msum = event_mask(t, *SUMMER_2023)
        in_span2 = bool(m2.any())
        y1 = m1.astype(int); y12 = (m1 | m2).astype(int)
        r = dict(protocol=proto, detector=det, n_rows=len(g), n_flags=n_flags,
                 alert_rate=float(f.mean()), in_test_span_tier1=bool(m1.any()),
                 in_test_span_tier2=in_span2)
        if n_flags:
            r.update(share_flags_in_tier1=float(f[m1].sum() / n_flags),
                     share_flags_in_tier2=float(f[m2].sum() / n_flags),
                     share_flags_jul_sep_2023=float(f[msum].sum() / n_flags),
                     share_flags_outside_labels=float(f[~(m1 | m2)].sum() / n_flags),
                     n_flags_jul_sep_2023=int(f[msum].sum()))
        for tag, y in (("tier1", y1), ("tier1_plus_tier2", y12)):
            prf = timestamp_level_prf(y, f)
            r.update({f"{tag}_precision": prf["precision"], f"{tag}_recall": prf["recall"],
                      f"{tag}_f1": prf["f1"], f"{tag}_prevalence": float(y.mean())})
            if n_boot and y.sum() and n_flags:
                pt, lo, hi, nb = block_bootstrap_ci(t, lambda idx, y=y: _f1_on(y, f, idx),
                                                    n_boot=n_boot, seed=seed)
                r.update({f"{tag}_f1_lo": lo, f"{tag}_f1_hi": hi, f"{tag}_n_day_blocks": nb})
        for name, c in per_event_coverage(t, f, tier1).items():
            r[f"cov_{name}"] = c["coverage"]
        for name, c in per_event_coverage(t, f, tier2).items():
            r[f"cov_{name}"] = c["coverage"]
        rows.append(r)
    return pd.DataFrame(rows)


def main(a):
    flags = pd.read_csv(a.flags)
    out = tier_report(flags, n_boot=a.n_boot, seed=a.seed)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    out.to_csv(a.out, index=False)
    pd.set_option("display.width", 200)
    show = ["protocol", "detector", "n_flags", "share_flags_in_tier2", "share_flags_jul_sep_2023",
            "share_flags_outside_labels", "tier1_f1", "tier1_plus_tier2_f1"]
    print(out[[c for c in show if c in out.columns]].round(3).to_string(index=False))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--flags", required=True, help="long CSV: protocol, detector, time, flag")
    ap.add_argument("--out", default="results/real/ercot_tiers.csv")
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    main(ap.parse_args())
