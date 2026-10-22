"""Classical detectors written to mirror the published Studies 2-4 pipeline, plus the
inductive / threshold-free variants the reviewers asked for.  All fit on TRAIN only
unless explicitly marked transductive."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from .windows import make_windows


def features(x, samples_per_day, window_days=7):
    """value, first difference ('return'), trailing rolling mean and std over a window fixed in
    TIME (window_days), so the feature meaning does not change with resolution."""
    s = pd.Series(x)
    w = max(3, int(window_days * samples_per_day))
    f = pd.DataFrame({
        "value": s,
        "ret": s.diff(),
        "roll_mean": s.rolling(w, min_periods=w // 2).mean(),
        "roll_std": s.rolling(w, min_periods=w // 2).std(),
    })
    return f.bfill().fillna(0.0)


def top_pct_flag(score, pct=95):
    """Published rule: flag the top (100-pct)% of TEST scores.  Alert budget fixed at 5%."""
    return (score > np.percentile(score, pct)).astype(int)


def run_classical(F_train, F_test, seed=0, pct=95, cols=None):
    """Returns {name: (score, flag)} on the test split.
      isolation_forest   train-fit, score=anomaly score, flag=top5%   (published)
      dbscan_published   FIT ON TEST (transductive), eps = p90 of 5-NN distance on train,
                         flag = noise points, so the alert rate floats with density (published)
      knn_inductive      score = distance to 5th nearest TRAIN neighbour; flag=top5% (inductive DBSCAN-like)
      z_published        |return| > its 95th percentile on TEST (what the code actually does)
      z_rolling          |value - trailing mean| / trailing std > 3   (what the paper text says)
    """
    cols = list(cols) if cols is not None else list(F_train.columns)
    sc = StandardScaler().fit(F_train[cols])
    Xtr, Xte = sc.transform(F_train[cols]), sc.transform(F_test[cols])
    out = {}
    iso = IsolationForest(n_estimators=100, contamination=0.05, random_state=seed, n_jobs=1).fit(Xtr)
    s = -iso.score_samples(Xte)
    out["isolation_forest"] = (s, top_pct_flag(s, pct))

    k = 5
    nn = NearestNeighbors(n_neighbors=k).fit(Xtr)
    d_tr, _ = nn.kneighbors(Xtr)
    eps = float(np.percentile(np.sort(d_tr[:, -1]), 90))
    db = DBSCAN(eps=eps, min_samples=5).fit(Xte)
    flag = (db.labels_ == -1).astype(int)
    out["dbscan_published"] = (flag.astype(float), flag)

    d_te, _ = nn.kneighbors(Xte, n_neighbors=k)
    s = d_te[:, -1]
    out["knn_inductive"] = (s, top_pct_flag(s, pct))

    r = np.abs(np.asarray(F_test["ret"]))
    out["z_published"] = (r, (r > np.percentile(r, pct)).astype(int))

    zr = np.abs((F_test["value"] - F_test["roll_mean"]) / (F_test["roll_std"] + 1e-9)).to_numpy()
    out["z_rolling"] = (zr, (zr > 3).astype(int))
    return out


def pca_window_reconstruction(x_train, t_train, x_test, t_test, seq_len=30, n_components=8,
                              pct=95):
    """Linear stand-in for a window autoencoder: PCA fitted on TRAIN windows; the score at the
    window's last timestamp is the squared reconstruction error.  Windows never span gaps.
    NOT a substitute for the LSTM-AE / Transformer -- only for the controlled sweep, where it
    plays the 'windowed reconstruction model' role."""
    Wtr, _ = make_windows(np.asarray(x_train)[:, None], t_train, seq_len)
    Wte, idx = make_windows(np.asarray(x_test)[:, None], t_test, seq_len)
    Wtr = Wtr.reshape(len(Wtr), -1); Wte = Wte.reshape(len(Wte), -1)
    pca = PCA(n_components=min(n_components, Wtr.shape[1]), random_state=0).fit(Wtr)
    rec = pca.inverse_transform(pca.transform(Wte))
    err = ((Wte - rec) ** 2).mean(1)
    full = np.full(len(x_test), np.nan); full[idx] = err
    return full, top_pct_flag(err, pct), idx
