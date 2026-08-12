"""
Trains an LSTM Autoencoder for sequence-based behavioral anomaly detection,
using the CERT Insider Threat email dataset (email.csv).

This is the second half of the Hybrid Autoencoder + LSTM architecture from
your proposal. The feedforward Autoencoder (train_autoencoder_local.py)
already handles point-in-time transaction fraud on creditcard.csv. This
script handles the *sequential* / behavioral side: it looks at a user's
recent stream of activity (not a single event) and learns what a "normal"
sequence looks like, the same way the Autoencoder learns what a normal
transaction looks like.

METHOD (same reconstruction-based logic as the Autoencoder, over time
instead of over features):
  1. Build per-user chronological sequences of events from email.csv.
  2. Train an LSTM Autoencoder to reconstruct those sequences.
  3. Sequences that don't compress/reconstruct well (high MSE) are
     behaviorally unusual -- e.g. a burst of large attachments sent to
     external domains outside someone's normal pattern.
  4. No ground-truth insider labels exist in email.csv, so -- honestly,
     same as you'd write in the report -- this is UNSUPERVISED. The
     threshold is chosen as a percentile of the reconstruction error
     distribution (95th by default), not by maximizing F1 against labels.

RUN THIS ON YOUR OWN MACHINE, with email.csv at dataset/cert/email.csv
(same path clean_email.py already expects).

Setup:
    pip install torch pandas numpy scikit-learn joblib

Usage:
    python train_lstm_local.py

Output (same folder):
    - lstm_autoencoder.pt        (trained PyTorch model weights)
    - lstm_scaler.pkl            (StandardScaler fit on sequence features)
    - lstm_threshold.json        (threshold + feature/sequence config)
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

DATA_PATH = "dataset/cert/email.csv"
MODEL_OUT = "lstm_autoencoder.pt"
SCALER_OUT = "lstm_scaler.pkl"
THRESHOLD_OUT = "lstm_threshold.json"

SEQ_LEN = 15          # events per sequence (a "behavioral window")
STRIDE = 5            # step between windows for the same user -- more training samples
FEATURE_DIM = 8
HIDDEN_DIM = 32
BOTTLENECK_DIM = 16
BATCH_SIZE = 64
MAX_EPOCHS = 60
PATIENCE = 8
THRESHOLD_PERCENTILE = 95
RANDOM_STATE = 42

FEATURE_NAMES = [
    "hour_of_day", "day_of_week", "minutes_since_last_norm",
    "email_size_norm", "attachments_norm", "num_recipients_norm",
    "is_external_recipient", "is_large_attachment",
]


# =========================================================
# Feature engineering -- shared logic, also used by lstm_inference.py
# =========================================================
def _domain(addr):
    if not isinstance(addr, str) or "@" not in addr:
        return ""
    return addr.split("@")[-1].strip().lower()


def event_to_features(row, prev_ts):
    """Turns one raw email event (dict-like) into an 8-dim feature vector."""
    ts = row.get("timestamp") or row.get("date")
    ts = pd.to_datetime(ts)

    minutes_since_last = 0.0
    if prev_ts is not None:
        minutes_since_last = max((ts - prev_ts).total_seconds() / 60.0, 0.0)

    size = float(row.get("size", row.get("email_size", 0)) or 0)
    attachments = float(row.get("attachments", 0) or 0)

    to_field = str(row.get("to", row.get("email_to", "")) or "")
    from_field = str(row.get("from", row.get("email_from", "")) or "")
    recipients = [r for r in to_field.replace(",", ";").split(";") if r.strip()]
    num_recipients = len(recipients)

    sender_domain = _domain(from_field)
    is_external = 0.0
    if sender_domain:
        for r in recipients:
            if _domain(r) and _domain(r) != sender_domain:
                is_external = 1.0
                break

    return [
        ts.hour / 23.0,
        ts.dayofweek / 6.0,
        np.log1p(minutes_since_last) / 10.0,       # log-compressed, roughly 0-1
        np.log1p(size) / 15.0,
        min(attachments / 5.0, 1.0),
        min(num_recipients / 10.0, 1.0),
        is_external,
        1.0 if attachments >= 3 else 0.0,
    ], ts


def build_sequences(df):
    """Groups events by user, sorts chronologically, and slices into
    overlapping fixed-length windows. Returns (N, SEQ_LEN, FEATURE_DIM) array
    plus parallel metadata (user, start_timestamp) for each window."""
    df = df.sort_values("date")
    sequences, meta = [], []

    for user, group in df.groupby("user"):
        group = group.sort_values("date")
        feats = []
        prev_ts = None
        for _, row in group.iterrows():
            vec, ts = event_to_features(row, prev_ts)
            feats.append(vec)
            prev_ts = ts

        if len(feats) < SEQ_LEN:
            continue

        for start in range(0, len(feats) - SEQ_LEN + 1, STRIDE):
            window = feats[start:start + SEQ_LEN]
            sequences.append(window)
            meta.append({"user": user, "window_start_idx": start})

    return np.array(sequences, dtype=np.float32), meta


# =========================================================
# Model
# =========================================================
class LSTMAutoencoder(nn.Module):
    def __init__(self, feature_dim=FEATURE_DIM, hidden_dim=HIDDEN_DIM,
                 bottleneck_dim=BOTTLENECK_DIM, seq_len=SEQ_LEN):
        super().__init__()
        self.seq_len = seq_len
        self.encoder_lstm = nn.LSTM(feature_dim, hidden_dim, batch_first=True)
        self.to_bottleneck = nn.Linear(hidden_dim, bottleneck_dim)
        self.from_bottleneck = nn.Linear(bottleneck_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, feature_dim)

    def forward(self, x):
        _, (h_n, _) = self.encoder_lstm(x)
        bottleneck = self.to_bottleneck(h_n[-1])
        decoder_input = self.from_bottleneck(bottleneck).unsqueeze(1)
        decoder_input = decoder_input.repeat(1, self.seq_len, 1)
        decoded, _ = self.decoder_lstm(decoder_input)
        return self.output_layer(decoded)


def reconstruction_error(model, X, device):
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32).to(device)
        recon = model(X_t)
        err = torch.mean((X_t - recon) ** 2, dim=(1, 2))
        return err.cpu().numpy()


def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Couldn't find '{DATA_PATH}'. Edit DATA_PATH at the top of "
            f"this file, or make sure email.csv is at dataset/cert/email.csv."
        )

    print("Loading CERT email dataset...")
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    print(f"Loaded {len(df)} email events from {df['user'].nunique()} users")

    print("Building per-user sequences...")
    X, meta = build_sequences(df)
    print(f"Built {len(X)} sequences of length {SEQ_LEN} "
          f"({FEATURE_DIM} features each)")

    if len(X) < 20:
        raise RuntimeError(
            "Not enough sequences to train on. You likely have very few "
            "users with >= SEQ_LEN events -- lower SEQ_LEN/STRIDE and retry."
        )

    # Scale features (fit on flattened sequences, apply per-timestep)
    scaler = StandardScaler()
    N, T, F = X.shape
    scaler.fit(X.reshape(-1, F))
    X_scaled = scaler.transform(X.reshape(-1, F)).reshape(N, T, F)

    X_train, X_val = train_test_split(X_scaled, test_size=0.2, random_state=RANDOM_STATE)
    print(f"Train sequences: {len(X_train)} | Validation sequences: {len(X_val)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    model = LSTMAutoencoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32)),
        batch_size=BATCH_SIZE, shuffle=True
    )

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    print("\nTraining LSTM Autoencoder...")
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for (batch,) in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch)
        train_loss = total_loss / len(X_train)

        val_errors = reconstruction_error(model, X_val, device)
        val_loss = float(np.mean(val_errors))

        print(f"Epoch {epoch:3d} | train_loss={train_loss:.5f} | val_loss={val_loss:.5f}")

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)")
                break

    model.load_state_dict(best_state)

    # Threshold: percentile of validation reconstruction error (unsupervised --
    # no insider-threat labels available in email.csv to optimize F1 against)
    val_errors = reconstruction_error(model, X_val, device)
    threshold = float(np.percentile(val_errors, THRESHOLD_PERCENTILE))

    print(f"\nSelected threshold ({THRESHOLD_PERCENTILE}th percentile of "
          f"validation reconstruction error): {threshold:.6f}")
    print(f"This flags the top ~{100 - THRESHOLD_PERCENTILE}% most unusual "
          f"behavioral sequences as anomalous.")

    torch.save(model.state_dict(), MODEL_OUT)
    joblib.dump(scaler, SCALER_OUT)
    with open(THRESHOLD_OUT, "w") as f:
        json.dump({
            "threshold": threshold,
            "threshold_percentile": THRESHOLD_PERCENTILE,
            "seq_len": SEQ_LEN,
            "feature_dim": FEATURE_DIM,
            "hidden_dim": HIDDEN_DIM,
            "bottleneck_dim": BOTTLENECK_DIM,
            "feature_names": FEATURE_NAMES,
            "val_reconstruction_error": {
                "mean": float(np.mean(val_errors)),
                "std": float(np.std(val_errors)),
                "min": float(np.min(val_errors)),
                "max": float(np.max(val_errors)),
            },
            "note": (
                "Unsupervised model -- email.csv has no ground-truth insider "
                "labels, so this threshold is a distributional cutoff, not an "
                "F1-optimized one. Report this honestly in the Evaluation "
                "Framework section, same as noted in fraud-inference.py."
            ),
        }, f, indent=2)

    print(f"\nSaved model to {MODEL_OUT}")
    print(f"Saved scaler to {SCALER_OUT}")
    print(f"Saved threshold + config to {THRESHOLD_OUT}")
    print("\nNext: run evaluate_lstm_model.py to generate report-ready charts,")
    print("then wire lstm_inference.py into main.py for live scoring.")


if __name__ == "__main__":
    main()