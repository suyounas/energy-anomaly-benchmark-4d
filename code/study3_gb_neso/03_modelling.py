# ═══════════════════════════════════════════════════════════════════════
# STUDY 3 — Phase 3: Model training + evaluation on REAL GB demand data.
# Methods, architectures, hyperparameters IDENTICAL to Study 2 — the
# four-dataset comparison is only valid if the pipeline is held constant.
#
# Additions (diagnostics only, core metrics unchanged):
#   - Per-event recall breakdown (collapse vs spike — opposite signs)
#   - "Beast from the East" check: what share of false positives fall in
#     Feb 15 - Mar 15 2018, a REAL unlabelled GB cold event? Flags there
#     are arguably correct detections, not errors.
#
# RUNTIME WARNING: this is the largest dataset yet (98k training rows
# under Protocol A). On Colab CPU this will be slow — enable the free
# GPU first: Runtime > Change runtime type > T4 GPU. Methods unchanged.
# ═══════════════════════════════════════════════════════════════════════

from google.colab import drive
drive.mount('/content/drive')

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score

DATA_DIR = '/content/drive/MyDrive/neso_data'
os.chdir(DATA_DIR)
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}' + ('' if device.type == 'cuda' else
      '  ⚠️ CPU — expect a long run; enable GPU for speed'))

FEATURES = ['return', 'roll_mean_30d', 'roll_std_30d', 'roll_skew_30d']
SEQ_LEN = 30  # rows, identical to Study 2 for method consistency

GROUND_TRUTH_EVENTS = [
    ("2020-03-23", "2020-05-31", "COVID collapse"),
    ("2022-12-08", "2022-12-17", "Cold snap spike"),
]
BEAST_WINDOW = ("2018-02-15", "2018-03-15")  # real, unlabelled GB event

# %% [1] Model classes — verbatim from Study 2 ------------------------------
class SeqDataset(Dataset):
    def __init__(self, df, feature_cols, scaler, seq_len=SEQ_LEN):
        X = scaler.transform(df[feature_cols])
        self.seqs = np.array([X[i:i+seq_len] for i in range(len(X) - seq_len)],
                              dtype=np.float32)
    def __len__(self):
        return len(self.seqs)
    def __getitem__(self, idx):
        return torch.tensor(self.seqs[idx], dtype=torch.float32)

class LSTMAutoencoder(nn.Module):
    def __init__(self, n_features, hidden_dim=32, latent_dim=8):
        super().__init__()
        self.encoder = nn.LSTM(n_features, hidden_dim, batch_first=True)
        self.to_latent = nn.Linear(hidden_dim, latent_dim)
        self.from_latent = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.LSTM(hidden_dim, n_features, batch_first=True)
    def forward(self, x):
        _, (h, _) = self.encoder(x)
        z = self.to_latent(h[-1])
        h_dec = self.from_latent(z).unsqueeze(1).repeat(1, x.size(1), 1)
        out, _ = self.decoder(h_dec)
        return out

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerAutoencoder(nn.Module):
    def __init__(self, n_features, d_model=32, nhead=4, num_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                dim_feedforward=64, batch_first=True, dropout=0.1)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, n_features)
    def forward(self, x):
        h = self.pos_enc(self.input_proj(x))
        seq_len = x.size(1)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        h = self.encoder(h, mask=mask)
        return self.output_proj(h)

def train_lstm_ae(train_ds, n_features, epochs=30, lr=1e-3, batch_size=32):
    model = LSTMAutoencoder(n_features).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    model.train()
    for epoch in range(epochs):
        for batch in loader:
            batch = batch.to(device)
            opt.zero_grad()
            loss = loss_fn(model(batch), batch)
            loss.backward()
            opt.step()
    return model

def train_transformer(train_ds, n_features, epochs=30, lr=5e-4, batch_size=32):
    model = TransformerAutoencoder(n_features).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.MSELoss()
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    model.train()
    for epoch in range(epochs):
        for batch in loader:
            batch = batch.to(device)
            opt.zero_grad()
            loss = loss_fn(model(batch), batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
    return model

def get_scores(model, ds, batch_size=64):
    model.eval()
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    scores = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            recon = model(batch)
            err = ((recon - batch) ** 2).mean(dim=[1, 2])
            scores.extend(err.cpu().numpy().tolist())
    return np.array(scores)

def get_classical(train_df, test_df):
    scaler = StandardScaler().fit(train_df[FEATURES])
    X_train = scaler.transform(train_df[FEATURES])
    X_test = scaler.transform(test_df[FEATURES])
    iso = IsolationForest(contamination=0.05, random_state=SEED).fit(X_train)
    iso_scores = -iso.score_samples(X_test)
    knn = NearestNeighbors(n_neighbors=5).fit(X_train)
    distances, _ = knn.kneighbors(X_train)
    eps = np.percentile(np.sort(distances[:, -1]), 90)
    db = DBSCAN(eps=eps, min_samples=5).fit(X_test)
    db_flag = (db.labels_ == -1).astype(int)
    z_flag = (test_df['return'].abs() > test_df['return'].abs().quantile(0.95)).astype(int)
    return iso_scores, db_flag, z_flag.values

def flag_from_scores(scores, pct=95):
    return (scores > np.percentile(scores, pct)).astype(int)

# %% [2] Evaluation with per-event breakdown --------------------------------
def per_event_recall(dates, y_true, flag):
    out = {}
    for start, end, name in GROUND_TRUTH_EVENTS:
        mask = (dates >= start) & (dates <= end)
        if mask.sum() == 0:
            out[name] = None
            continue
        out[name] = flag[mask].mean()  # fraction of event rows flagged
    return out

def beast_fp_share(dates, y_true, flag):
    fp_mask = (flag == 1) & (y_true == 0)
    n_fp = fp_mask.sum()
    if n_fp == 0:
        return 0.0, 0
    in_beast = fp_mask & (dates >= BEAST_WINDOW[0]) & (dates <= BEAST_WINDOW[1])
    return in_beast.sum() / n_fp, n_fp

def run_pipeline(train_df, test_df, label):
    print(f'\n{"="*70}\n{label}\n{"="*70}')
    scaler = StandardScaler().fit(train_df[FEATURES])
    train_ds = SeqDataset(train_df, FEATURES, scaler)
    test_ds = SeqDataset(test_df, FEATURES, scaler)

    print('Training LSTM Autoencoder...')
    lstm_scores = get_scores(train_lstm_ae(train_ds, len(FEATURES)), test_ds)
    print('Training Transformer Encoder...')
    trans_scores = get_scores(train_transformer(train_ds, len(FEATURES)), test_ds)
    iso_scores, db_flag, z_flag = get_classical(train_df, test_df)

    y_seq = test_df['is_anomaly'].values[SEQ_LEN:]
    y_full = test_df['is_anomaly'].values
    d_seq = test_df['Date'].values[SEQ_LEN:]
    d_full = test_df['Date'].values

    rows = []
    for name, flag, y, dates in [
        ('LSTM Autoencoder', flag_from_scores(lstm_scores), y_seq, d_seq),
        ('Transformer Encoder', flag_from_scores(trans_scores), y_seq, d_seq),
        ('Isolation Forest', flag_from_scores(iso_scores), y_full, d_full),
        ('DBSCAN', db_flag, y_full, d_full),
        ('Rolling Z-score', z_flag, y_full, d_full),
    ]:
        p = precision_score(y, flag, zero_division=0)
        r = recall_score(y, flag, zero_division=0)
        f1 = f1_score(y, flag, zero_division=0)
        ev = per_event_recall(pd.to_datetime(dates), y, flag)
        beast_share, n_fp = beast_fp_share(pd.to_datetime(dates), y, flag)
        ev_str = '  '.join(f'{k}: {v:.2f}' if v is not None else f'{k}: n/a'
                            for k, v in ev.items())
        print(f'  {name:<22} P={p:.3f} R={r:.3f} F1={f1:.3f} | {ev_str} | '
              f'FPs in Beast window: {beast_share:.1%} of {n_fp}')
        rows.append({'Method': name, 'Precision': round(p,3), 'Recall': round(r,3),
                     'F1': round(f1,3),
                     **{f'Recall_{k}': (round(v,3) if v is not None else None)
                        for k, v in ev.items()},
                     'FP_share_in_Beast_window': round(beast_share,3)})
    return rows

# %% [3] Run both protocols --------------------------------------------------
trainA = pd.read_csv('neso_trainA.csv', parse_dates=['Date'])
testA = pd.read_csv('neso_testA.csv', parse_dates=['Date'])
trainB = pd.read_csv('neso_trainB.csv', parse_dates=['Date'])
testB = pd.read_csv('neso_testB.csv', parse_dates=['Date'])

rows_A = run_pipeline(trainA, testA,
    'GB DEMAND — Protocol A (forward 70/30; COVID in training = contaminated)')
rows_B = run_pipeline(trainB, testB,
    'GB DEMAND — Protocol B (clean window; both events reachable)')

out = []
for r in rows_A:
    out.append({'Dataset': 'GB Demand (NESO, real)', 'Protocol': 'A (forward 70/30)', **r})
for r in rows_B:
    out.append({'Dataset': 'GB Demand (NESO, real)', 'Protocol': 'B (clean window)', **r})
final = pd.DataFrame(out)
final.to_csv('neso_phase3_results.csv', index=False)
print('\n' + '=' * 70)
print('FINAL — saved neso_phase3_results.csv')
print('=' * 70)
print(final[['Protocol','Method','Precision','Recall','F1']].to_string(index=False))
