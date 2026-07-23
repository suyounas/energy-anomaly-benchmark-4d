# ═══════════════════════════════════════════════════════════════════════
# Unsupervised Anomaly Detection in Energy Markets — REAL DATA, REAL MODELS
# Run this in Google Colab: colab.research.google.com -> File -> New notebook
# Paste this whole file into one cell (or split at the "# %%" markers into
# separate cells) and Runtime -> Run all.
# No local setup needed. Free CPU is fine; GPU (Runtime -> Change runtime
# type -> T4 GPU) will make LSTM/Transformer training faster but isn't required.
# ═══════════════════════════════════════════════════════════════════════

# %% [1] Install / imports -------------------------------------------------
!pip install -q pandas-datareader

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import requests, io

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# %% [2] Pull REAL data directly from FRED (public, no API key needed) -----
def fetch_fred_series(series_id):
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    df.columns = ["Date", series_id]
    df["Date"] = pd.to_datetime(df["Date"])
    df[series_id] = pd.to_numeric(df[series_id], errors="coerce")
    return df.dropna().reset_index(drop=True)

wti = fetch_fred_series("DCOILWTICO")     # WTI crude oil, $/barrel
hh  = fetch_fred_series("DHHNGSP")        # Henry Hub natural gas, $/MMBtu

print(f"WTI: {len(wti)} real observations, {wti['Date'].min().date()} to {wti['Date'].max().date()}")
print(f"Henry Hub: {len(hh)} real observations, {hh['Date'].min().date()} to {hh['Date'].max().date()}")

# Restrict to a common, sufficiently long window (adjust as you like)
START, END = "2010-01-01", "2024-12-31"
wti = wti[(wti.Date >= START) & (wti.Date <= END)].reset_index(drop=True)
hh  = hh[(hh.Date >= START) & (hh.Date <= END)].reset_index(drop=True)

# %% [3] Feature engineering (identical treatment for both series) --------
def build_features(df, price_col):
    df = df.copy()
    df["return"] = df[price_col].pct_change()
    df["roll_mean_30"] = df[price_col].rolling(30, min_periods=5).mean()
    df["roll_std_30"]  = df[price_col].rolling(30, min_periods=5).std()
    df["roll_skew_30"] = df["return"].rolling(30, min_periods=5).skew()
    df["zscore"] = (df[price_col] - df["roll_mean_30"]) / df["roll_std_30"].replace(0, np.nan)
    df = df.dropna().reset_index(drop=True)
    return df

wti_f = build_features(wti, "DCOILWTICO")
hh_f  = build_features(hh, "DHHNGSP")

FEATURES = ["return", "roll_mean_30", "roll_std_30", "roll_skew_30"]

# %% [4] Documented ground-truth anomaly windows (for evaluation only —
# never used during training, preserving the unsupervised principle) ------
GROUND_TRUTH_EVENTS = [
    ("2014-11-01", "2016-02-29"),  # OPEC price war collapse
    ("2020-03-01", "2020-04-30"),  # COVID demand collapse (incl. negative WTI)
    ("2021-02-10", "2021-02-25"),  # Texas deep freeze (Henry Hub spike)
    ("2022-02-20", "2022-06-30"),  # Russia-Ukraine energy crisis
]

def label_ground_truth(df):
    df = df.copy()
    df["is_anomaly"] = 0
    for start, end in GROUND_TRUTH_EVENTS:
        mask = (df.Date >= start) & (df.Date <= end)
        df.loc[mask, "is_anomaly"] = 1
    return df

wti_f = label_ground_truth(wti_f)
hh_f  = label_ground_truth(hh_f)

# %% [5] Train/test split — chronological, no lookahead --------------------
def chrono_split(df, train_frac=0.7):
    n = len(df)
    cut = int(n * train_frac)
    return df.iloc[:cut].reset_index(drop=True), df.iloc[cut:].reset_index(drop=True)

wti_train, wti_test = chrono_split(wti_f)
hh_train, hh_test   = chrono_split(hh_f)

scaler_wti = StandardScaler().fit(wti_train[FEATURES])
scaler_hh  = StandardScaler().fit(hh_train[FEATURES])

# %% [6] Sequence dataset for the neural models -----------------------------
SEQ_LEN = 30

class SeqDataset(Dataset):
    def __init__(self, df, feature_cols, scaler, seq_len=SEQ_LEN):
        X = scaler.transform(df[feature_cols])
        self.seqs = []
        for i in range(len(X) - seq_len):
            self.seqs.append(X[i:i+seq_len])
        self.seqs = np.array(self.seqs, dtype=np.float32)

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        x = self.seqs[idx]
        return torch.tensor(x, dtype=torch.float32)

# %% [7] REAL LSTM Autoencoder ----------------------------------------------
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
        total_loss = 0
        for batch in loader:
            batch = batch.to(device)
            opt.zero_grad()
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            opt.step()
            total_loss += loss.item() * batch.size(0)
        if (epoch + 1) % 10 == 0:
            print(f"  LSTM-AE epoch {epoch+1}/{epochs}  loss={total_loss/len(train_ds):.5f}")
    return model

def lstm_ae_scores(model, ds, batch_size=64):
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

# %% [8] REAL Transformer Encoder (built from scratch, no pretrained weights)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerAutoencoder(nn.Module):
    def __init__(self, n_features, d_model=32, nhead=4, num_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=64,
            batch_first=True, dropout=0.1
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, n_features)

    def forward(self, x):
        h = self.input_proj(x)
        h = self.pos_enc(h)
        # causal mask so each position only attends to itself and the past
        seq_len = x.size(1)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        h = self.encoder(h, mask=mask)
        return self.output_proj(h)

def train_transformer(train_ds, n_features, epochs=30, lr=5e-4, batch_size=32):
    model = TransformerAutoencoder(n_features).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.MSELoss()
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in loader:
            batch = batch.to(device)
            opt.zero_grad()
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item() * batch.size(0)
        sched.step()
        if (epoch + 1) % 10 == 0:
            print(f"  Transformer epoch {epoch+1}/{epochs}  loss={total_loss/len(train_ds):.5f}")
    return model

def transformer_scores(model, ds, batch_size=64):
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

# %% [9] Classical baselines (Isolation Forest, DBSCAN, rolling z-score) ---
def classical_scores(train_df, test_df, feature_cols, scaler):
    X_train = scaler.transform(train_df[feature_cols])
    X_test = scaler.transform(test_df[feature_cols])

    iso = IsolationForest(contamination=0.05, random_state=SEED).fit(X_train)
    iso_scores = -iso.score_samples(X_test)  # higher = more anomalous

    db = DBSCAN(eps=1.5, min_samples=5).fit(X_test)
    db_scores = (db.labels_ == -1).astype(float)  # 1 = noise/anomaly

    z_scores = test_df["zscore"].abs().fillna(0).values
    z_flag = (z_scores > 3).astype(float)

    return iso_scores, db_scores, z_flag

# %% [10] Run the full pipeline on one commodity ----------------------------
def run_pipeline(train_df, test_df, name):
    print(f"\n{'='*60}\nRunning pipeline for {name}\n{'='*60}")
    scaler = StandardScaler().fit(train_df[FEATURES])

    train_ds = SeqDataset(train_df, FEATURES, scaler)
    test_ds  = SeqDataset(test_df, FEATURES, scaler)

    print("Training LSTM Autoencoder (real PyTorch training, ~1-2 min)...")
    lstm_model = train_lstm_ae(train_ds, n_features=len(FEATURES))
    lstm_sc = lstm_ae_scores(lstm_model, test_ds)

    print("Training Transformer Encoder (real PyTorch training, ~2-3 min)...")
    trans_model = train_transformer(train_ds, n_features=len(FEATURES))
    trans_sc = transformer_scores(trans_model, test_ds)

    iso_sc, db_sc, z_sc = classical_scores(train_df, test_df, FEATURES, scaler)

    # align: sequence-based scores are offset by SEQ_LEN vs classical scores
    y_true_seq = test_df["is_anomaly"].values[SEQ_LEN:]
    y_true_flat = test_df["is_anomaly"].values

    def threshold_and_score(scores, y_true, pct=95):
        thresh = np.percentile(scores, pct)
        y_pred = (scores > thresh).astype(int)
        p = precision_score(y_true, y_pred, zero_division=0)
        r = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        return p, r, f1

    results = {}
    results["LSTM Autoencoder"]    = threshold_and_score(lstm_sc, y_true_seq)
    results["Transformer Encoder"] = threshold_and_score(trans_sc, y_true_seq)
    results["Isolation Forest"]    = threshold_and_score(iso_sc, y_true_flat)
    results["DBSCAN"]              = (
        precision_score(y_true_flat, db_sc, zero_division=0),
        recall_score(y_true_flat, db_sc, zero_division=0),
        f1_score(y_true_flat, db_sc, zero_division=0),
    )
    results["Rolling Z-score"] = (
        precision_score(y_true_flat, z_sc, zero_division=0),
        recall_score(y_true_flat, z_sc, zero_division=0),
        f1_score(y_true_flat, z_sc, zero_division=0),
    )

    print(f"\n{'Method':<22}{'Precision':<12}{'Recall':<12}{'F1':<10}")
    for k, (p, r, f1) in results.items():
        print(f"{k:<22}{p:<12.3f}{r:<12.3f}{f1:<10.3f}")

    return results

wti_results = run_pipeline(wti_train, wti_test, "WTI Crude Oil")
hh_results  = run_pipeline(hh_train, hh_test, "Henry Hub Natural Gas")

# %% [11] Save results to CSV for the paper/repo ----------------------------
def results_to_df(results, commodity):
    rows = []
    for method, (p, r, f1) in results.items():
        rows.append({"Commodity": commodity, "Method": method,
                      "Precision": round(p, 3), "Recall": round(r, 3), "F1": round(f1, 3)})
    return pd.DataFrame(rows)

final = pd.concat([
    results_to_df(wti_results, "WTI"),
    results_to_df(hh_results, "Henry Hub"),
])
final.to_csv("real_model_comparison_results.csv", index=False)
print("\nSaved: real_model_comparison_results.csv")
print(final.to_string(index=False))

# %% [12] Download the results file to your machine -------------------------
from google.colab import files
files.download("real_model_comparison_results.csv")
