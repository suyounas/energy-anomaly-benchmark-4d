# ═══════════════════════════════════════════════════════════════════════
# STUDY 4 — Diagnostic: why did LSTM/Transformer score exactly 0 on
# ERCOT when LSTM led on GB (F1=0.282)?
#
# H-A (dilution): 5-day events barely move 30-day rolling features
#     (~5/30 of amplitude), so events are near-invisible in this feature
#     space to reconstruction models; GB's 10-week COVID dwarfed the
#     window and was visible. Test: per-feature z-scores of event rows.
# H-B (drift): ERCOT demand grew secularly 2019-2023; NNs trained on
#     2021-22 (Protocol B) may spend the entire 5% flag budget on 2019
#     (low-demand era) before any event can rank. Test: monthly
#     histogram of where LSTM flags actually land.
# ═══════════════════════════════════════════════════════════════════════

from google.colab import drive
drive.mount('/content/drive')

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

DATA_DIR = '/content/drive/MyDrive/eia_data'
os.chdir(DATA_DIR)
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

FEATURES = ['return', 'roll_mean_30d', 'roll_std_30d', 'roll_skew_30d']
SEQ_LEN = 30

trainB = pd.read_csv('ercot_trainB.csv', parse_dates=['Date'])
testB = pd.read_csv('ercot_testB.csv', parse_dates=['Date'])

scaler = StandardScaler().fit(trainB[FEATURES])
X_test = scaler.transform(testB[FEATURES])

# ═══════════════════════════════════════════════════════════════════════
# %% [H-A] Event visibility in the feature space (no training needed)
# ═══════════════════════════════════════════════════════════════════════
print('\n' + '=' * 70)
print('H-A: per-feature z-scores of event rows (vs trainB distribution)')
print('=' * 70)

windows = {
    'NORMAL (all non-event)': (testB['is_anomaly'] == 0),
    'Texas Freeze': (testB.Date >= '2021-02-14') & (testB.Date <= '2021-02-19'),
    'Elliott': (testB.Date >= '2022-12-22') & (testB.Date <= '2022-12-26'),
    'Jul-22 heat (unlabelled)': (testB.Date >= '2022-07-10') & (testB.Date <= '2022-07-25'),
    '2019 (drift check)': (testB.Date < '2020-01-01'),
}

knn = NearestNeighbors(n_neighbors=5).fit(scaler.transform(trainB[FEATURES]))

for label, mask in windows.items():
    Xm = X_test[mask.values]
    if len(Xm) == 0:
        continue
    zs = '  '.join(f'{f}={Xm[:, i].mean():+.2f}' for i, f in enumerate(FEATURES))
    d, _ = knn.kneighbors(Xm)
    print(f'\n{label} ({mask.sum()} rows)')
    print(f'  mean z: {zs}')
    print(f'  5-NN dist to train: median={np.median(d[:, -1]):.3f}')

print('\nDilution arithmetic (expected, for reference): a 5-day event at')
print('+40% amplitude shifts a 30-day rolling mean by roughly 40%*(5/30)')
print('= ~6-7% at its peak — check whether roll_mean_30d z above for the')
print('events is correspondingly small vs what GB COVID showed (z=-1.25).')

# ═══════════════════════════════════════════════════════════════════════
# %% [H-B] Where do LSTM flags actually land? (one retrain, small data)
# ═══════════════════════════════════════════════════════════════════════
print('\n' + '=' * 70)
print('H-B: monthly distribution of LSTM flags on testB')
print('=' * 70)

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

train_ds = SeqDataset(trainB, FEATURES, scaler)
test_ds = SeqDataset(testB, FEATURES, scaler)
print('Training LSTM (same seed/hyperparameters as Phase 3)...')
model = train_lstm_ae(train_ds, len(FEATURES))

model.eval()
loader = DataLoader(test_ds, batch_size=64, shuffle=False)
scores = []
with torch.no_grad():
    for batch in loader:
        batch = batch.to(device)
        err = ((model(batch) - batch) ** 2).mean(dim=[1, 2])
        scores.extend(err.cpu().numpy().tolist())
scores = np.array(scores)
flag = (scores > np.percentile(scores, 95)).astype(int)

dates_seq = pd.to_datetime(testB['Date'].values[SEQ_LEN:])
flagged_dates = dates_seq[flag.astype(bool)]

print(f'\nTotal flags: {flag.sum()}')
print('Flags by year-month (top 15):')
ym = pd.Series(flagged_dates).dt.to_period('M').value_counts().sort_values(ascending=False)
for period, count in ym.head(15).items():
    print(f'  {period}: {count}')

# Score ranking of event rows: how anomalous does the LSTM think they are?
for label in ['Texas Freeze', 'Elliott']:
    mask = windows[label]
    seq_mask = mask.values[SEQ_LEN:]
    if seq_mask.sum():
        ranks = pd.Series(scores).rank(pct=True)[seq_mask]
        print(f'\n{label}: LSTM score percentile of event rows — '
              f'median {ranks.median():.1%}, max {ranks.max():.1%}')
        print(f'  (flag threshold is the 95.0% percentile — rows below it can never be flagged)')

print('\nDiagnostic complete. Send full output for interpretation.')
