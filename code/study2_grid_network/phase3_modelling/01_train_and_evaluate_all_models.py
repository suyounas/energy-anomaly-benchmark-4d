# ═══════════════════════════════════════════════════════════════════════
# Phase 3b: Model training + evaluation. Reuses the proven LSTM/Transformer/
# IsolationForest/DBSCAN pipeline from the WTI/Henry Hub work, applied here
# to: (1) France load data under BOTH split protocols, for sensitivity
# comparison, and (2) France line-flow data (nuclear crisis) under Protocol B.
# ═══════════════════════════════════════════════════════════════════════

from google.colab import drive
drive.mount('/content/drive')

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
import os

DATA_DIR = '/content/drive/MyDrive/grid_data'
os.chdir(DATA_DIR)
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

FEATURES = ['return', 'roll_mean_30d', 'roll_std_30d', 'roll_skew_30d']
SEQ_LEN = 30  # NOTE: this is now 30 ROWS of hourly-derived features, not
              # 30 days directly — rows already reflect 30-day rolling stats

GROUND_TRUTH_EVENTS = [
    ("2016-10-10", "2017-01-27", "French nuclear crisis"),
    ("2020-03-16", "2020-05-11", "COVID-19 demand collapse"),
]

# %% [1] Model classes (identical to the WTI/Henry Hub proven versions) -----
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

def get_classical_scores(train_df, test_df, feature_cols):
    scaler = StandardScaler().fit(train_df[feature_cols])
    X_train = scaler.transform(train_df[feature_cols])
    X_test = scaler.transform(test_df[feature_cols])

    iso = IsolationForest(contamination=0.05, random_state=SEED).fit(X_train)
    iso_scores = -iso.score_samples(X_test)

    knn = NearestNeighbors(n_neighbors=5).fit(X_train)
    distances, _ = knn.kneighbors(X_train)
    eps = np.percentile(np.sort(distances[:, -1]), 90)
    db = DBSCAN(eps=eps, min_samples=5).fit(X_test)
    db_flag = (db.labels_ == -1).astype(float)

    z_flag = (test_df['return'].abs() > test_df['return'].abs().quantile(0.95)).astype(float) \
        if 'return' in test_df else np.zeros(len(test_df))

    return iso_scores, db_flag, z_flag

def flag_from_scores(scores, pct=95):
    return (scores > np.percentile(scores, pct)).astype(int)

# %% [2] Full pipeline runner -------------------------------------------------
def run_pipeline(train_df, test_df, label, value_col_features=FEATURES):
    print(f'\n{"="*70}\n{label}\n{"="*70}')
    scaler = StandardScaler().fit(train_df[FEATURES])
    train_ds = SeqDataset(train_df, FEATURES, scaler)
    test_ds = SeqDataset(test_df, FEATURES, scaler)

    print('Training LSTM Autoencoder...')
    lstm_model = train_lstm_ae(train_ds, n_features=len(FEATURES))
    lstm_scores = get_scores(lstm_model, test_ds)

    print('Training Transformer Encoder...')
    trans_model = train_transformer(train_ds, n_features=len(FEATURES))
    trans_scores = get_scores(trans_model, test_ds)

    iso_scores, db_flag, z_flag = get_classical_scores(train_df, test_df, FEATURES)

    y_true_seq = test_df['is_anomaly'].values[SEQ_LEN:]
    y_true_full = test_df['is_anomaly'].values

    lstm_flag = flag_from_scores(lstm_scores)
    trans_flag = flag_from_scores(trans_scores)
    iso_flag = flag_from_scores(iso_scores)

    results = {}
    for name, flag, y_true in [
        ('LSTM Autoencoder', lstm_flag, y_true_seq),
        ('Transformer Encoder', trans_flag, y_true_seq),
        ('Isolation Forest', iso_flag, y_true_full),
        ('DBSCAN', db_flag.astype(int), y_true_full),
        ('Rolling Z-score', z_flag.astype(int), y_true_full),
    ]:
        p = precision_score(y_true, flag, zero_division=0)
        r = recall_score(y_true, flag, zero_division=0)
        f1 = f1_score(y_true, flag, zero_division=0)
        results[name] = (p, r, f1)
        print(f'  {name:<22} P={p:.3f}  R={r:.3f}  F1={f1:.3f}')

    return results

# %% [3] Run on FRANCE LOAD DATA, both protocols -----------------------------
trainA = pd.read_csv('train_FR.csv', parse_dates=['Date'])
testA = pd.read_csv('test_FR.csv', parse_dates=['Date'])
results_A = run_pipeline(trainA, testA, 'FRANCE LOAD — Protocol A (forward 70/30 split)')

trainB = pd.read_csv('trainB_FR.csv', parse_dates=['Date'])
testB = pd.read_csv('testB_FR.csv', parse_dates=['Date'])
results_B = run_pipeline(trainB, testB, 'FRANCE LOAD — Protocol B (clean training window)')

# %% [4] Run on LINE-FLOW DATA (nuclear crisis), Protocol B only ------------
train_line = pd.read_csv('trainB_lineflow.csv', parse_dates=['Date'])
test_line = pd.read_csv('testB_lineflow.csv', parse_dates=['Date'])
results_line = run_pipeline(train_line, test_line, 'FRANCE LINE FLOW — Protocol B (nuclear crisis)')

# %% [5] Combined results table + save ---------------------------------------
rows = []
for method, (p, r, f1) in results_A.items():
    rows.append({'Dataset': 'FR Load', 'Protocol': 'A (forward 70/30)', 'Method': method,
                 'Precision': round(p,3), 'Recall': round(r,3), 'F1': round(f1,3)})
for method, (p, r, f1) in results_B.items():
    rows.append({'Dataset': 'FR Load', 'Protocol': 'B (clean window)', 'Method': method,
                 'Precision': round(p,3), 'Recall': round(r,3), 'F1': round(f1,3)})
for method, (p, r, f1) in results_line.items():
    rows.append({'Dataset': 'FR Line Flow', 'Protocol': 'B (clean window)', 'Method': method,
                 'Precision': round(p,3), 'Recall': round(r,3), 'F1': round(f1,3)})

final_df = pd.DataFrame(rows)
final_df.to_csv('phase3_final_results.csv', index=False)
print('\n' + '='*70)
print('FINAL COMBINED RESULTS')
print('='*70)
print(final_df.to_string(index=False))
print('\nSaved: phase3_final_results.csv')
