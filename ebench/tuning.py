"""Equal-budget hyper-parameter search with validation events SEPARATE from test events.

Design (answers W2 / R3-3 / register row 10 and 17):
  * every detector family gets the same number of random-search trials (`n_trials`) drawn from a
    small, declared search space, the same selection metric (validation AUC-PR) and the same
    number of final seeds;
  * the validation set is the LAST part of the clean training window with SYNTHETIC events
    injected (shift + burst, several durations).  The real test events are never seen;
  * the default (untuned) configuration is always evaluated too and reported, so the gain from
    tuning is visible and a detector cannot look good only because it was tuned harder.
"""
from __future__ import annotations
import numpy as np, pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from .metrics import auc_pr
from .synthetic import inject_events, STEPS_PER_DAY


# ------------------------------------------------------------ scorers (classical)
def _std(Ftr, Fev, cols):
    sc = StandardScaler().fit(Ftr[cols])
    return sc.transform(Ftr[cols]), sc.transform(Fev[cols])


def score_isolation_forest(cfg, Ftr, Fev, cols, seed, **_):
    Xtr, Xev = _std(Ftr, Fev, cols)
    m = IsolationForest(n_estimators=cfg["n_estimators"], max_samples=cfg["max_samples"],
                        random_state=seed, n_jobs=1).fit(Xtr)
    return -m.score_samples(Xev)


def score_knn(cfg, Ftr, Fev, cols, seed, **_):
    Xtr, Xev = _std(Ftr, Fev, cols)
    nn = NearestNeighbors(n_neighbors=cfg["k"]).fit(Xtr)
    return nn.kneighbors(Xev, n_neighbors=cfg["k"])[0][:, -1]


def score_rolling_z(cfg, Ftr, Fev, cols, seed, spd=48, **_):
    w = max(8, int(cfg["window_days"] * spd))
    allr = pd.concat([Ftr["ret"], Fev["ret"]])
    mu = allr.shift(1).rolling(w, min_periods=w // 4).mean()
    sd = allr.shift(1).rolling(w, min_periods=w // 4).std()
    z = ((allr - mu).abs() / (sd + 1e-12)).iloc[len(Ftr):]
    return z.fillna(0).to_numpy()


def score_neural(kind):
    def f(cfg, Ftr, Fev, cols, seed, **kw):
        from .neural import reconstruction_scores
        sc = StandardScaler().fit(Ftr[cols])
        return reconstruction_scores(kind, cfg, sc.transform(Ftr[cols]), Ftr.index,
                                     sc.transform(Fev[cols]), Fev.index, seed)
    return f


FAMILIES = {
    "isolation_forest": dict(fn=score_isolation_forest,
        default=dict(n_estimators=100, max_samples="auto"),
        space=dict(n_estimators=[50, 100, 200, 400], max_samples=["auto", 256, 1024])),
    "knn_inductive": dict(fn=score_knn, default=dict(k=5), space=dict(k=[3, 5, 10, 20, 50])),
    "rolling_z": dict(fn=score_rolling_z, default=dict(window_days=7),
        space=dict(window_days=[1, 3, 7, 14, 30])),
    "lstm_ae": dict(fn=score_neural("lstm_ae"),
        default=dict(hidden=32, latent=8, lr=1e-3, epochs=30),
        space=dict(hidden=[16, 32, 64], latent=[4, 8, 16], lr=[1e-3, 5e-4, 2e-3], epochs=[10, 20, 30])),
    "transformer": dict(fn=score_neural("transformer"),
        default=dict(d_model=32, nhead=4, layers=2, lr=5e-4, epochs=30),
        space=dict(d_model=[16, 32, 64], nhead=[2, 4], layers=[1, 2, 3], lr=[5e-4, 1e-3, 2e-3],
                   epochs=[10, 20, 30])),
}


def sample_configs(space, n, rng):
    keys = list(space)
    seen, out, tries = set(), [], 0
    while len(out) < n and tries < 50 * n:
        tries += 1
        cfg = {k: space[k][int(rng.integers(len(space[k])))] for k in keys}
        key = tuple(sorted(cfg.items(), key=lambda kv: kv[0]))
        if key not in seen:
            seen.add(key); out.append(cfg)
    return out


# ------------------------------------------------------------ validation set
def inject_into_series(t, x, val_start_idx, durations_days, prevalence, magnitude_sd, seed, base_minutes=30):
    """Inject synthetic events into x[val_start_idx:], alternating shift/burst across durations.
    Operates on the base 30-min grid (snap = 1 day); returns x_new, y."""
    rng = np.random.default_rng(seed)
    x_new = np.asarray(x, float).copy(); y = np.zeros(len(x), int)
    kinds = ["shift", "burst"]
    n_test = len(x) - val_start_idx
    per_cfg = prevalence / (len(durations_days) * len(kinds))
    # carve the validation region into equal blocks, one per (duration, kind)
    combos = [(d, k) for d in durations_days for k in kinds]
    block = n_test // len(combos)
    for i, (d, k) in enumerate(combos):
        s0 = val_start_idx + i * block
        sub_x = x_new[s0:s0 + block]
        spd = int(24 * 60 / base_minutes)
        if block < 4 * d * spd:
            continue
        x2, y2, _ = inject_events(sub_x, t[s0:s0 + block], 0, d * spd, per_cfg, magnitude_sd, rng, k,
                                  scale=float(np.std(x[:val_start_idx])))
        x_new[s0:s0 + block] = x2; y[s0:s0 + block] = y2
    return x_new, y



# ------------------------------------------------------------ validation suite (taxonomy)
SUITE_TYPES = ("shift", "burst", "ramp", "pulse")
PULSE_STEPS = 6                       # 3 h on the 30-min grid: survives resampling up to factor 6


def suite_cells(types=SUITE_TYPES, durations_days=(1, 3, 7), magnitudes=(1.5,)):
    """The declared event taxonomy: one 'cell' per (type, duration, magnitude).  Fixed a priori from
    the kinds of grid stress the paper targets (level shifts, variance bursts, drifts, short
    pulses) -- NOT from the test events -- so validation never sees test labels.  'pulse' has no
    duration (always 3 h)."""
    cells = []
    for ty in types:
        for d in ([0] if ty == "pulse" else durations_days):
            for m in magnitudes:
                cells.append(dict(cell=len(cells), type=ty, duration_days=d, magnitude=m))
    return pd.DataFrame(cells)


def make_validation_suite(t, x, val_start_idx, base_minutes=30, durations_days=(1, 3, 7),
                          magnitudes=(1.5,), types=SUITE_TYPES, days_per_cell=4.0, seed=0):
    """Inject the taxonomy into x[val_start_idx:] on the NATIVE grid.  The validation region is cut
    into equal slots, one per cell, so every cell has its own events and its own clean surroundings.
    Returns x_new, cell (int per timestamp, -1 = clean) and the cell table.  Magnitudes are in units
    of the clean-part standard deviation of x."""
    rng = np.random.default_rng(seed)
    spd = int(24 * 60 / base_minutes)
    tbl = suite_cells(types, durations_days, magnitudes)
    x_new = np.asarray(x, float).copy(); cell = np.full(len(x), -1, int)
    scale = float(np.std(x[:val_start_idx]))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("injection scale is not finite/positive")
    slot = (len(x) - val_start_idx) // len(tbl)
    for i, r in tbl.iterrows():
        lo = val_start_idx + i * slot
        L = PULSE_STEPS if r.type == "pulse" else int(r.duration_days * spd)
        n_ev = 4 if r.type == "pulse" else max(1, int(round(days_per_cell / r.duration_days)))
        if slot < n_ev * (L + 2 * spd):
            raise ValueError(f"validation region too short for cell {r.to_dict()}; "
                             f"use a longer window or fewer cells")
        stride = slot // n_ev
        for k in range(n_ev):
            room = stride - L - spd                                  # keep >= 1 clean day after the event
            s = lo + k * stride + int(rng.integers(0, max(1, room)))
            if r.type != "pulse":
                s -= (s - val_start_idx) % spd                       # day-aligned like the test events
                s = max(s, lo + k * stride)
            e = s + L
            amp = r.magnitude * scale
            if r.type == "shift":
                x_new[s:e] += amp
            elif r.type == "burst":
                x_new[s:e] += amp * rng.standard_normal(e - s)
            elif r.type == "ramp":
                x_new[s:e] += amp * np.linspace(0.0, 1.0, e - s)
            elif r.type == "pulse":
                x_new[s:e] += amp
            cell[s:e] = int(r.cell)
    return x_new, cell, tbl


def macro_lift(score, cell, n_cells):
    """Per-cell AUC-PR lift (positives = that cell's event timestamps, negatives = clean timestamps
    only, so another cell's events never count as false alarms) and its mean over cells.  A lift of 1
    is chance.  NaN scores are dropped.  Returns (mean_lift, per_cell_lifts)."""
    score = np.asarray(score, float); cell = np.asarray(cell)
    ok = ~np.isnan(score); score, cell = score[ok], cell[ok]
    clean = cell < 0
    lifts = []
    for c in range(n_cells):
        m = clean | (cell == c)
        pos = (cell[m] == c).astype(int)
        if pos.sum() == 0 or pos.sum() == len(pos):
            lifts.append(np.nan); continue
        lifts.append(auc_pr(pos, score[m]) / pos.mean())
    return float(np.nanmean(lifts)), lifts


# ------------------------------------------------------------ the search
def equal_budget_search(families, Ftr, Fval, yval, cols, n_trials=8, seed=0, spd=48, min_spread=0.02,
                        cells=None, n_cells=0, min_lift=1.3):
    """For each family: evaluate the default config and `n_trials` random configs on the
    validation set; select by validation AUC-PR.  Returns a tidy DataFrame (all trials) and the
    chosen config per family."""
    if int(np.sum(yval)) == 0:
        raise ValueError("validation set contains no injected events; tuning would be meaningless")
    rng = np.random.default_rng(seed)
    rows, chosen = [], {}
    for name in families:
        fam = FAMILIES[name]
        cands = [("default", fam["default"])] + [("random", c) for c in sample_configs(fam["space"], n_trials, rng)]
        best = (-1.0, None); start = len(rows)
        prev = float(np.mean(yval))
        for kind, cfg in cands:
            sc = fam["fn"](cfg, Ftr, Fval, cols, seed, spd=spd)
            ok = ~np.isnan(sc)
            val = auc_pr(yval[ok], sc[ok])
            extra = {}
            if cells is not None:
                ml, per = macro_lift(sc, cells, n_cells)
                extra = dict(val_macro_lift=ml, **{f"lift_cell_{i}": v for i, v in enumerate(per)})
                crit = ml
            else:
                crit = val
            rows.append(dict(family=name, trial_kind=kind, val_auc_pr=val, val_prevalence=prev,
                             val_lift=val / prev if prev else np.nan, **extra,
                             **{f"cfg_{k}": v for k, v in cfg.items()}))
            if not np.isnan(crit) and crit > best[0]:
                best = (crit, cfg)
        if best[1] is None:
            raise ValueError(f"{name}: every validation AUC-PR was NaN; refusing to fall back silently")
        key = "val_macro_lift" if cells is not None else "val_auc_pr"
        vals = np.array([r[key] for r in rows[start:]], float)
        spread = float(np.nanmax(vals) - np.nanmin(vals))
        informative = spread >= min_spread and (cells is None or float(np.nanmax(vals)) >= min_lift)
        chosen[name] = best[1] if informative else fam["default"]
        for r in rows[start:]:
            r["val_spread"] = spread; r["informative"] = informative
    return pd.DataFrame(rows), chosen
