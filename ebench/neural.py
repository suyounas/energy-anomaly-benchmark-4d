"""LSTM autoencoder and Transformer-encoder reconstruction scorers (PyTorch).

Architecture follows the published Study 1-4 scripts (LSTM decoder maps hidden->features directly;
Transformer feed-forward width 64) except that the positional encoding is a learned parameter
here rather than sinusoidal; so numbers from this module are NOT byte-identical to the published tables.

UNTESTED IN THE AUTHORING SANDBOX (no torch available there).  Intended to be run on the
user's Colab/GPU; `experiments/equal_budget_tuning.py --families ... lstm_ae transformer`.
torch is imported lazily so the rest of ebench works without it.
"""
from __future__ import annotations
import numpy as np

from .windows import make_windows


def _torch():
    import torch, torch.nn as nn   # noqa: F401
    return torch, nn


def _build_lstm_ae(nn, n_feat, hidden, latent):
    class LSTMAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = nn.LSTM(n_feat, hidden, batch_first=True)
            self.to_lat = nn.Linear(hidden, latent)
            self.from_lat = nn.Linear(latent, hidden)
            self.dec = nn.LSTM(hidden, n_feat, batch_first=True)   # as in the published scripts

        def forward(self, x):
            _, (h, _) = self.enc(x)
            z = self.to_lat(h[-1])
            rep = self.from_lat(z).unsqueeze(1).repeat(1, x.size(1), 1)
            y, _ = self.dec(rep)
            return y
    return LSTMAE()


def _build_transformer(nn, n_feat, d_model, nhead, layers, seq_len, ff=64):
    import torch

    class TE(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Linear(n_feat, d_model)
            self.pos = nn.Parameter(torch.zeros(1, seq_len, d_model))
            layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=ff,
                                               batch_first=True, dropout=0.1)
            self.enc = nn.TransformerEncoder(layer, layers)
            self.out = nn.Linear(d_model, n_feat)

        def forward(self, x):
            return self.out(self.enc(self.inp(x) + self.pos))
    return TE()


def reconstruction_scores(kind, cfg, X_fit, t_fit, X_eval, t_eval, seed, seq_len=30, device=None):
    """Train on windows of X_fit (segment-aware), return per-timestamp score on X_eval
    (squared reconstruction error of the window ending at that timestamp; NaN before the first
    full window of each contiguous segment)."""
    torch, nn = _torch()
    torch.manual_seed(seed); np.random.seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    Wf, _ = make_windows(X_fit, t_fit, seq_len)
    We, idx = make_windows(X_eval, t_eval, seq_len)
    n_feat = Wf.shape[-1]
    if kind == "lstm_ae":
        model = _build_lstm_ae(nn, n_feat, cfg["hidden"], cfg["latent"])
        opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    elif kind == "transformer":
        model = _build_transformer(nn, n_feat, cfg["d_model"], cfg["nhead"], cfg["layers"], seq_len)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
    else:
        raise ValueError(kind)
    model.to(device)
    ds = torch.tensor(Wf, dtype=torch.float32)
    loader = torch.utils.data.DataLoader(ds, batch_size=cfg.get("batch", 64), shuffle=True)
    lossf = nn.MSELoss()
    for _ in range(cfg["epochs"]):
        model.train()
        for b in loader:
            b = b.to(device); opt.zero_grad()
            loss = lossf(model(b), b); loss.backward(); opt.step()
    model.eval(); errs = []
    with torch.no_grad():
        for i in range(0, len(We), 512):
            b = torch.tensor(We[i:i + 512], dtype=torch.float32, device=device)
            errs.append(((model(b) - b) ** 2).mean((1, 2)).cpu().numpy())
    full = np.full(len(X_eval), np.nan)
    if errs:
        full[idx] = np.concatenate(errs)
    return full
