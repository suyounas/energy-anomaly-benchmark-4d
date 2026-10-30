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


# ------------------------------------------------------------ the search
def equal_budget_search(families, Ftr, Fval, yval, cols, n_trials=8, seed=0, spd=48, min_spread=0.02):
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
            rows.append(dict(family=name, trial_kind=kind, val_auc_pr=val, val_prevalence=prev,
                             val_lift=val / prev if prev else np.nan,
                             **{f"cfg_{k}": v for k, v in cfg.items()}))
            if not np.isnan(val) and val > best[0]:
                best = (val, cfg)
        if best[1] is None:
            raise ValueError(f"{name}: every validation AUC-PR was NaN; refusing to fall back silently")
        vals = np.array([r["val_auc_pr"] for r in rows[start:]], float)
        spread = float(np.nanmax(vals) - np.nanmin(vals))
        informative = spread >= min_spread
        chosen[name] = best[1] if informative else fam["default"]
        for r in rows[start:]:
            r["val_spread"] = spread; r["informative"] = informative
    return pd.DataFrame(rows), chosen
