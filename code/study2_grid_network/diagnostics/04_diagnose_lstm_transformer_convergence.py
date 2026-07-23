# ═══════════════════════════════════════════════════════════════════════
# Self-contained version — retrains LSTM + Transformer fresh on line-flow
# data rather than assuming Phase 3b's session state is still around.
# ═══════════════════════════════════════════════════════════════════════

from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import os

DATA_DIR = '/content/drive/MyDrive/grid_data'
os.chdir(DATA_DIR)
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

FEATURES = ['return', 'roll_mean_30d', 'roll_std_30d', 'roll_skew_30d']
SEQ_LEN = 30

# %% [1] Model classes (same as Phase 3b) ------------------------------------
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

def flag_from_scores(scores, pct=95):
    return (scores > np.percentile(scores, pct)).astype(int)

# %% [2] Load line-flow train/test data (already prepared by Phase 3a) -----
required_files = ['trainB_lineflow.csv', 'testB_lineflow.csv']
missing = [f for f in required_files if not os.path.exists(f)]
if missing:
    raise RuntimeError(f'Missing: {missing}. Re-run Phase 3a first.')

train_line = pd.read_csv('trainB_lineflow.csv', parse_dates=['Date'])
test_line = pd.read_csv('testB_lineflow.csv', parse_dates=['Date'])
print(f'Loaded train_line ({len(train_line)} rows), test_line ({len(test_line)} rows)')

# %% [3] Retrain both models fresh -------------------------------------------
scaler_line = StandardScaler().fit(train_line[FEATURES])
train_ds_line = SeqDataset(train_line, FEATURES, scaler_line)
test_ds_line = SeqDataset(test_line, FEATURES, scaler_line)

print('Training LSTM Autoencoder...')
lstm_model = train_lstm_ae(train_ds_line, n_features=len(FEATURES))
lstm_scores_line = get_scores(lstm_model, test_ds_line)

print('Training Transformer Encoder...')
trans_model = train_transformer(train_ds_line, n_features=len(FEATURES))
trans_scores_line = get_scores(trans_model, test_ds_line)

# %% [4] The actual diagnostic ------------------------------------------------
print('\n' + '=' * 70)
print('DIAGNOSTIC: identical scores, or just identical flags at threshold?')
print('=' * 70)

print(f'\nRaw score arrays identical? {np.array_equal(lstm_scores_line, trans_scores_line)}')
print(f'LSTM score range: min={lstm_scores_line.min():.4f}, max={lstm_scores_line.max():.4f}')
print(f'Transformer score range: min={trans_scores_line.min():.4f}, max={trans_scores_line.max():.4f}')
print(f'Correlation between the two raw score arrays: '
      f'{np.corrcoef(lstm_scores_line, trans_scores_line)[0,1]:.4f}')

lstm_flag_line = flag_from_scores(lstm_scores_line)
trans_flag_line = flag_from_scores(trans_scores_line)

print(f'\nFlagged (binary) arrays identical? {np.array_equal(lstm_flag_line, trans_flag_line)}')
print(f'LSTM flagged count: {lstm_flag_line.sum()}')
print(f'Transformer flagged count: {trans_flag_line.sum()}')
print(f'Overlap (both flag the same row): {(lstm_flag_line & trans_flag_line).sum()}')
print(f'LSTM-only flags: {(lstm_flag_line & ~trans_flag_line).sum()}')
print(f'Transformer-only flags: {(~lstm_flag_line & trans_flag_line).sum()}')

dates_seq_line = test_line['Date'].values[SEQ_LEN:]
lstm_flagged_dates = pd.to_datetime(dates_seq_line[lstm_flag_line.astype(bool)])
trans_flagged_dates = pd.to_datetime(dates_seq_line[trans_flag_line.astype(bool)])

print(f'\nLSTM flagged date range: {lstm_flagged_dates.min()} to {lstm_flagged_dates.max()}')
print(f'Transformer flagged date range: {trans_flagged_dates.min()} to {trans_flagged_dates.max()}')

print('\n' + '=' * 70)
print('INTERPRETATION')
print('=' * 70)
if np.array_equal(lstm_scores_line, trans_scores_line):
    print('⚠️ Raw scores are BIT-IDENTICAL — this strongly suggests a bug')
    print('(e.g. one model object accidentally aliased to the other).')
elif np.array_equal(lstm_flag_line, trans_flag_line):
    print('Raw scores DIFFER, but the binary flagged set is identical.')
    print('This is a real (if striking) convergence — both architectures')
    print('rank the same days as most anomalous, even with different')
    print('continuous error magnitudes. Worth reporting as a genuine finding,')
    print('framed accurately as "identical flagged sets", not "independent')
    print('agreement".')
else:
    print('Flagged sets differ — worth re-examining the original P/R/F1')
    print('identity as likely a metric-level coincidence.')
