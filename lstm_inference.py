"""
Loads the trained LSTM Autoencoder and scores a user's recent event
sequence for behavioral anomalies.

This is the live-serving counterpart to train_lstm_local.py -- same
feature encoding, same model class, so a sequence built from MongoDB
live events is scored exactly the same way as the training data was.

IMPORTANT LIMITATION (be upfront about this in your report, same as
fraud-inference.py is upfront about its own limitation):
The model was trained on CERT email.csv behavioral patterns (send/receive
volume, attachments, external recipients, timing). It generalizes best to
email-type events. For non-email SIEM events (logins, admin actions, etc.)
several features (size, attachments, recipients) will simply be 0 --
the model can still pick up on *timing* anomalies (unusual hour, bursts)
for those, but it wasn't trained on that behavior specifically. Frame the
live dashboard demo as "sequence-based behavioral scoring", not as a
fully-validated cross-event-type detector.

Put this file in the same folder as lstm_autoencoder.pt, lstm_scaler.pkl,
and lstm_threshold.json (the three files train_lstm_local.py produces).

Usage from main.py:
    from lstm_inference import score_user_sequence

    events = list(logs_collection.find({"user_id": user_id})
                   .sort("timestamp", -1).limit(15))
    result = score_user_sequence(events)
    # result = {"available": bool, "behavior_score": float, "is_anomaly": bool, ...}
"""

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
from pathlib import Path

_DIR = Path(__file__).parent

_model = None
_scaler = None
_meta = None
_device = torch.device("cpu")  # inference is cheap; no need for GPU in the API


class LSTMAutoencoder(nn.Module):
    """Must match the architecture in train_lstm_local.py exactly."""
    def __init__(self, feature_dim, hidden_dim, bottleneck_dim, seq_len):
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


def _domain(addr):
    if not isinstance(addr, str) or "@" not in addr:
        return ""
    return addr.split("@")[-1].strip().lower()


def _event_to_features(event, prev_ts):
    """Same encoding as train_lstm_local.py's event_to_features, but reads
    field names as they appear on live SIEM events (LogEvent / Mongo docs)
    OR raw CERT rows -- whichever keys are present."""
    ts_raw = event.get("timestamp") or event.get("date")
    ts = pd.to_datetime(ts_raw) if ts_raw else pd.Timestamp.utcnow()

    minutes_since_last = 0.0
    if prev_ts is not None:
        minutes_since_last = max((ts - prev_ts).total_seconds() / 60.0, 0.0)

    size = float(event.get("email_size", event.get("size", 0)) or 0)
    attachments = float(event.get("attachments", 0) or 0)

    to_field = str(event.get("email_to", event.get("to", "")) or "")
    from_field = str(event.get("email_from", event.get("from", "")) or "")
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
        np.log1p(minutes_since_last) / 10.0,
        np.log1p(size) / 15.0,
        min(attachments / 5.0, 1.0),
        min(num_recipients / 10.0, 1.0),
        is_external,
        1.0 if attachments >= 3 else 0.0,
    ], ts


def _load():
    global _model, _scaler, _meta
    if _model is not None:
        return

    with open(_DIR / "lstm_threshold.json") as f:
        _meta = json.load(f)

    _model = LSTMAutoencoder(
        feature_dim=_meta["feature_dim"],
        hidden_dim=_meta["hidden_dim"],
        bottleneck_dim=_meta["bottleneck_dim"],
        seq_len=_meta["seq_len"],
    )
    _model.load_state_dict(torch.load(_DIR / "lstm_autoencoder.pt", map_location=_device))
    _model.eval()

    _scaler = joblib.load(_DIR / "lstm_scaler.pkl")


def build_sequence_features(events):
    """events: list of dicts, OLDEST FIRST. Returns a (seq_len, feature_dim)
    array, left-padded with zero-vectors if fewer than seq_len events exist."""
    _load()
    seq_len = _meta["seq_len"]

    feats = []
    prev_ts = None
    for e in events:
        vec, ts = _event_to_features(e, prev_ts)
        feats.append(vec)
        prev_ts = ts

    if len(feats) > seq_len:
        feats = feats[-seq_len:]
    elif len(feats) < seq_len:
        pad = [[0.0] * _meta["feature_dim"]] * (seq_len - len(feats))
        feats = pad + feats

    return np.array(feats, dtype=np.float32)


def score_user_sequence(events, min_events=5):
    """
    events: list of dicts (SIEM log docs or CERT rows), OLDEST FIRST, for
            a single user. Pass the last ~15 events for that user.
    min_events: below this many real events, scoring is unreliable (too much
                zero-padding) -- returns available=False instead of a
                misleadingly confident score.

    Returns:
        {
          "available": bool,
          "reconstruction_error": float,
          "behavior_score": float,   # 0-100, comparable in spirit to fraud_score
          "is_anomaly": bool,
          "threshold": float,
        }
    """
    _load()

    if len(events) < min_events:
        return {
            "available": False,
            "reason": f"Need at least {min_events} events for this user "
                      f"to score a behavioral sequence reliably; have {len(events)}.",
        }

    seq = build_sequence_features(events)
    T, F = seq.shape
    seq_scaled = _scaler.transform(seq.reshape(-1, F)).reshape(1, T, F)

    with torch.no_grad():
        x = torch.tensor(seq_scaled, dtype=torch.float32)
        recon = _model(x)
        error = torch.mean((x - recon) ** 2).item()

    threshold = _meta["threshold"]
    behavior_score = float(np.clip((error / threshold) * 50, 0, 100))

    return {
        "available": True,
        "reconstruction_error": error,
        "behavior_score": round(behavior_score, 1),
        "is_anomaly": error > threshold,
        "threshold": threshold,
    }


if __name__ == "__main__":
    # Quick smoke test on a slice of your CERT data
    import pandas as pd
    df = pd.read_csv("dataset/cert/email.csv")
    df["date"] = pd.to_datetime(df["date"])
    user = df["user"].value_counts().index[0]  # most active user
    user_events = df[df["user"] == user].sort_values("date").tail(15).to_dict("records")
    result = score_user_sequence(user_events)
    print(f"User: {user}")
    print(result)