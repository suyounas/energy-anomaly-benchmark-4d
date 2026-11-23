#!/usr/bin/env python3
"""Step 10: fetch EIA-930 hourly demand for ALL balancing authorities and apply the series-selection rule.

Uses the same public six-month bulk files as code/study4_ercot_eia (no API key).
Rule (docs/STEP10_PREREGISTRATION.md): a BA qualifies if, over WINDOW, at least 95% of hours have demand
and no gap is longer than 72 hours, after the same cleaning as the ERCOT pipeline (numeric parse, drop NaN,
drop duplicate timestamps). Demand = 'Demand (MW) (Adjusted)' where present, else 'Demand (MW)'.

Outputs in --out:
  eia930_demand_long.csv.gz   ba, datetime_utc, demand_mw          (all BAs, all window hours that exist)
  series_manifest.csv         ba, coverage, longest_gap_h, share_adjusted, qualifies
  raw_file_sha256.txt         hash of every downloaded bulk file
Raw files are deleted after filtering (hashes kept). Re-running resumes from per-file caches.
"""
import argparse, hashlib, os, sys
import pandas as pd
import requests

BASE = "https://www.eia.gov/electricity/gridmonitor/sixMonthFiles"
WIN_START, WIN_END = pd.Timestamp("2019-07-01 00:00"), pd.Timestamp("2026-07-31 23:00")
MIN_COVERAGE, MAX_GAP_H = 0.95, 72

def halves():
    for y in range(WIN_START.year, WIN_END.year + 1):
        for h, (a, b) in (("Jan_Jun", (1, 6)), ("Jul_Dec", (7, 12))):
            if pd.Timestamp(y, b, 1) + pd.offsets.MonthEnd(0) >= WIN_START and pd.Timestamp(y, a, 1) <= WIN_END:
                yield y, h

def col(cols, *needles, exclude=()):
    for n in needles:
        for c in cols:
            if n in c and not any(x in c for x in exclude):
                return c
    return None

def filter_file(path):
    """Stream a bulk file to a compact long frame: ba, datetime_utc, demand_adj, demand_raw."""
    out = []
    for ch in pd.read_csv(path, chunksize=200_000, low_memory=False):
        ba = col(ch.columns, "Balancing Authority")
        t = col(ch.columns, "UTC Time at End of Hour")
        adj = col(ch.columns, "Demand (MW) (Adjusted)")
        raw = next((c for c in ch.columns if c.strip() == "Demand (MW)"), None)
        if ba is None or t is None or (adj is None and raw is None):
            sys.exit(f"{path}: unexpected columns {list(ch.columns)[:10]}")
        d = pd.DataFrame({"ba": ch[ba], "t": ch[t],
                          "adj": ch[adj] if adj else None, "raw": ch[raw] if raw else None})
        out.append(d)
    return pd.concat(out, ignore_index=True)

def parse_time(s):
    """Fast path for the common EIA format 'MM/DD/YYYY HH:MM:SS AM'; falls back to the generic parser."""
    for fmt in ("%m/%d/%Y %I:%M:%S %p", "%m/%d/%Y %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            out = pd.to_datetime(s, format=fmt, errors="coerce")
        except Exception:
            continue
        if out.notna().mean() > 0.99:
            return out
    return pd.to_datetime(s, errors="coerce")

def to_num(s):
    return pd.to_numeric(s.astype(str).str.replace(",", "", regex=False), errors="coerce")

def clean(df):
    df = df.copy()
    df["datetime_utc"] = parse_time(df["t"])
    df["adj"], df["raw"] = to_num(df["adj"]), to_num(df["raw"])
    df["demand_mw"] = df["adj"].combine_first(df["raw"])
    df["used_adj"] = df["adj"].notna()
    df = df.dropna(subset=["datetime_utc", "demand_mw"])
    df = df[(df.datetime_utc >= WIN_START) & (df.datetime_utc <= WIN_END)]
    return df.sort_values(["ba", "datetime_utc"]).drop_duplicates(["ba", "datetime_utc"])

def longest_gap_hours(ts, start=None, end=None):
    idx = pd.date_range(start or WIN_START, end or WIN_END, freq="h")
    present = pd.Series(idx.isin(ts), index=idx)
    miss = (~present).astype(int)
    run = miss.groupby((miss != miss.shift()).cumsum()).cumsum() * miss
    return int(run.max()), int(len(idx))

def manifest(long):
    rows = []
    for ba, g in long.groupby("ba"):
        gap, n = longest_gap_hours(g["datetime_utc"])
        cov = len(g) / n
        rows.append(dict(ba=ba, coverage=round(cov, 4), longest_gap_h=gap, share_adjusted=round(g.used_adj.mean(), 3),
                         qualifies=bool(cov >= MIN_COVERAGE and gap <= MAX_GAP_H)))
    return pd.DataFrame(rows).sort_values("ba")

def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--local-dir", help="use EIA930_BALANCE_<year>_<half>.csv from here instead of downloading (testing)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    frames, hashes = [], []
    for y, h in halves():
        name = f"EIA930_BALANCE_{y}_{h}.csv"
        cache = os.path.join(a.out, f"cache_{y}_{h}.csv.gz")
        if os.path.exists(cache):
            print(y, h, "cached"); frames.append(pd.read_csv(cache)); continue
        tmp = os.path.join(a.out, "tmp_" + name)
        if a.local_dir:
            tmp = os.path.join(a.local_dir, name)
            if not os.path.exists(tmp):
                print(y, h, "missing locally, skipped"); continue
        else:
            print(y, h, "downloading", f"{BASE}/{name}")
            with requests.get(f"{BASE}/{name}", stream=True, timeout=900) as r:
                if r.status_code == 404:
                    print(y, h, "not published (404), skipped"); continue
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for c in r.iter_content(1 << 20):
                        f.write(c)
        hashes.append(f"{sha(tmp)}  {name}")
        d = filter_file(tmp)
        d.to_csv(cache, index=False, compression="gzip")
        frames.append(d)
        if not a.local_dir:
            os.remove(tmp)
    long = clean(pd.concat(frames, ignore_index=True))
    long[["ba", "datetime_utc", "demand_mw"]].to_csv(os.path.join(a.out, "eia930_demand_long.csv.gz"), index=False, compression="gzip")
    m = manifest(long)
    m.to_csv(os.path.join(a.out, "series_manifest.csv"), index=False)
    open(os.path.join(a.out, "raw_file_sha256.txt"), "w").write("\n".join(hashes) + "\n")
    print(m.qualifies.value_counts().to_string())
    print(f"{int(m.qualifies.sum())} of {len(m)} BAs qualify; manifest in {a.out}/series_manifest.csv")
    print("window end observed:", long.datetime_utc.max())

if __name__ == "__main__":
    main()
