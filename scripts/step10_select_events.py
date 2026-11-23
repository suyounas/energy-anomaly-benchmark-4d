#!/usr/bin/env python3
"""Step 10 event selection (source class B: NWS/NCEI Storm Events mapped to balancing authorities).

Run BEFORE any detector. Deterministic: same inputs -> same output. Rules are those in
docs/STEP10_PREREGISTRATION.md; change a rule there first, never here only.

Inputs (all public):
  --storm-dir   directory with StormEvents_details-ftp_v1.0_dYYYY_c*.csv.gz (NCEI bulk CSV)
  --territory   PUDL out_eia861__yearly_balancing_authority_service_territory (.parquet or .csv)
  --ba-codes    CSV with columns ba_id,ba_code mapping EIA balancing-authority IDs to EIA-930 codes
  --out         output events CSV
"""
import argparse, glob, json, re, sys
from collections import defaultdict
import pandas as pd

FAMILIES = {
    "heat": {"Excessive Heat", "Heat"},
    "cold": {"Extreme Cold/Wind Chill", "Winter Storm", "Blizzard", "Ice Storm"},
    "tropical": {"Hurricane", "Hurricane (Typhoon)", "Tropical Storm"},
}
MIN_ZONES_PER_STATE = 3      # distinct zones/counties with a qualifying event in the episode
MIN_BA_POP_SHARE = 0.5       # share of the BA's population living in covered states
MIN_DAYS, MAX_DAYS = 1, 21   # inclusive calendar days
START, END = pd.Timestamp("2019-07-01"), pd.Timestamp("2026-07-31")  # END: last month >= 3 months before the 2026-11-23 download
NEED = ["EPISODE_ID", "EVENT_TYPE", "STATE_FIPS", "CZ_TYPE", "CZ_FIPS", "BEGIN_DATE_TIME", "END_DATE_TIME"]

def load_storms(d):
    frames = []
    for f in sorted(glob.glob(f"{d}/StormEvents_details*_d20*.csv*")):
        yr = int(re.search(r"_d(\d{4})_c", f).group(1))
        if not (START.year <= yr <= END.year):
            continue
        df = pd.read_csv(f, usecols=lambda c: c in NEED, dtype=str, compression="infer")
        miss = [c for c in NEED if c not in df.columns]
        if miss:
            sys.exit(f"{f}: missing columns {miss}; check the NCEI format document and fix NEED")
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    fam = {t: k for k, v in FAMILIES.items() for t in v}
    df["family"] = df["EVENT_TYPE"].map(fam)
    df = df.dropna(subset=["family"]).copy()
    for c in ("BEGIN_DATE_TIME", "END_DATE_TIME"):
        df[c] = pd.to_datetime(df[c], format="%d-%b-%y %H:%M:%S", errors="coerce")
    df = df.dropna(subset=["BEGIN_DATE_TIME", "END_DATE_TIME"])
    df["STATE_FIPS"] = df["STATE_FIPS"].str.zfill(2)
    df["CZ_FIPS"] = df["CZ_TYPE"].fillna("") + df["CZ_FIPS"].fillna("")
    return df

def episode_states(df):
    g = df.groupby(["EPISODE_ID", "family", "STATE_FIPS"]).agg(
        zones=("CZ_FIPS", "nunique"), start=("BEGIN_DATE_TIME", "min"), end=("END_DATE_TIME", "max")).reset_index()
    g = g[g["zones"] >= MIN_ZONES_PER_STATE].copy()
    g["start"], g["end"] = g["start"].dt.normalize(), g["end"].dt.normalize()
    return g

def ba_state_pop(terr, ba_codes):
    terr = terr.assign(balancing_authority_id_eia=terr["balancing_authority_id_eia"].astype(str))
    t = terr.merge(ba_codes, left_on="balancing_authority_id_eia", right_on="ba_id", how="inner")
    t["year"] = pd.to_datetime(t["report_date"]).dt.year
    t["state_fips"] = t["state_id_fips"].astype(str).str.zfill(2)
    return t.groupby(["ba_code", "year", "state_fips"])["population"].sum().reset_index()

def pop_lookup(pop):
    d = defaultdict(dict)
    for r in pop.itertuples():
        d[(r.ba_code, r.year)][r.state_fips] = r.population
    years = defaultdict(list)
    for (b, y) in d:
        years[b].append(y)
    return d, {b: sorted(v) for b, v in years.items()}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--storm-dir", required=True); ap.add_argument("--territory", required=True)
    ap.add_argument("--ba-codes", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    es = episode_states(load_storms(a.storm_dir))
    terr = pd.read_csv(a.territory) if a.territory.endswith('.csv') else pd.read_parquet(a.territory)
    pop = ba_state_pop(terr, pd.read_csv(a.ba_codes, dtype=str))
    lut, yrs = pop_lookup(pop)
    rows = []
    for (ep, fam), grp in es.groupby(["EPISODE_ID", "family"]):
        y = int(grp["start"].min().year)
        for ba, ys in yrs.items():
            use = max([v for v in ys if v <= y], default=ys[0])
            sp = lut[(ba, use)]; tot = sum(sp.values())
            if tot <= 0:
                continue
            hit = grp[grp["STATE_FIPS"].isin(sp)]
            share = sum(sp[s] for s in hit["STATE_FIPS"].unique()) / tot
            if share >= MIN_BA_POP_SHARE:
                rows.append(dict(ba=ba, episode=ep, family=fam, start=hit["start"].min(), end=hit["end"].max(), share=round(share, 3)))
    ev = pd.DataFrame(rows)
    if ev.empty:
        sys.exit("no events selected")
    ev = ev[(ev.start >= START) & (ev.end <= END)]
    # merge overlapping or adjacent windows within a BA
    merged = []
    for ba, g in ev.sort_values(["ba", "start"]).groupby("ba"):
        cur = None
        for r in g.itertuples():
            if cur and r.start <= cur["end"] + pd.Timedelta(days=1):
                cur["end"] = max(cur["end"], r.end); cur["family"].add(r.family); cur["eps"].add(r.episode)
            else:
                if cur: merged.append(cur)
                cur = dict(ba=ba, start=r.start, end=r.end, family={r.family}, eps={r.episode})
        if cur: merged.append(cur)
    m = pd.DataFrame(merged)
    m["days"] = (m["end"] - m["start"]).dt.days + 1
    n_all = len(m)
    m = m[(m.days >= MIN_DAYS) & (m.days <= MAX_DAYS)].reset_index(drop=True)
    # storm groups: connected components over windows that overlap in time and share a family
    parent = list(range(len(m)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for i in range(len(m)):
        for j in range(i + 1, len(m)):
            if m.start[i] <= m.end[j] and m.start[j] <= m.end[i] and (m.family[i] & m.family[j]):
                parent[find(i)] = find(j)
    m["group"] = [find(i) for i in range(len(m))]
    gid = {g: f"B{k+1:03d}" for k, g in enumerate(sorted(set(m.group)))}
    out = pd.DataFrame(dict(
        event_id=[f"B{ i+1:04d}" for i in range(len(m))], storm_group=m.group.map(gid), source_class="B_nws_ncei",
        source_url="https://www.ncei.noaa.gov/stormevents/ftp.jsp", start_date=m.start.dt.date, end_date=m.end.dt.date,
        geography=m.family.map(lambda s: "+".join(sorted(s))), BAs_mapped=m.ba,
        mapping_rationale=[f"episodes {','.join(sorted(e))}; >= {MIN_ZONES_PER_STATE} zones per state; BA population share >= {MIN_BA_POP_SHARE}" for e in m.eps]))
    out.to_csv(a.out, index=False)
    summ = dict(events_before_duration_filter=int(n_all), events=int(len(out)), storm_groups=int(out.storm_group.nunique()),
                bas=int(out.BAs_mapped.nunique()))
    print(json.dumps(summ, indent=2))
    json.dump(summ, open(a.out.replace(".csv", "_summary.json"), "w"), indent=2)

if __name__ == "__main__":
    main()
