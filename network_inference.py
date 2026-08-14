"""
Loads the trained Network Intrusion Autoencoder and scores network-flow
records for anomalies. Same pattern as fraud_inference.py.

IMPORTANT LIMITATION (be upfront about this in your report, same as
fraud_inference.py and lstm_inference.py are upfront about their own):
This model was trained on UNSW-NB15's 45 flow-level features (packet
counts, byte counts, TCP window sizes, inter-packet timing, etc.) --
these come from a network flow capture tool (Argus/Bro-IDS), not from
your application layer. Your live SIEM events (login attempts, payments,
admin actions) don't naturally produce this feature set either -- same
honest gap the fraud model has with V1-V28.

Two ways to use this, same framing as fraud_inference.py:
  1. OFFLINE EVALUATION (recommended, defensible): score the official
     held-out UNSW-NB15 test set and report precision/recall/F1/AUPRC --
     see evaluate_network_intrusion_model.py.
  2. LIVE DEMO (simulated stream): replay held-out test-set rows through
     the live pipeline, clearly labelled as such -- see network_stream.py
     (mirrors fraud_stream.py).

Put this file in the same folder as network_autoencoder.pkl,
network_scaler.pkl, network_encoder_columns.pkl, and
network_threshold.json.
"""

import numpy as np
import pandas as pd
import joblib
import json
from pathlib import Path

_DIR = Path(__file__).parent

_model = None
_scaler = None
_threshold = None
_encoder_columns = None
_categorical_columns = None


def _load():
    global _model, _scaler, _threshold, _encoder_columns, _categorical_columns
    if _model is None:
        _model = joblib.load(_DIR / "network_autoencoder.pkl")
        _scaler = joblib.load(_DIR / "network_scaler.pkl")
        _encoder_columns = joblib.load(_DIR / "network_encoder_columns.pkl")
        with open(_DIR / "network_threshold.json") as f:
            meta = json.load(f)
        _threshold = meta["threshold"]
        _categorical_columns = meta["categorical_columns"]


def _encode(df):
    """One-hot encodes proto/service/state and aligns columns to the
    exact layout the model was trained on (fills anything missing with 0,
    drops anything extra) -- required since a small replay batch won't
    naturally contain every category the full training set did."""
    encoded = pd.get_dummies(df, columns=_categorical_columns)
    return encoded.reindex(columns=_encoder_columns, fill_value=0)


def explain_flow(row, top_n=5):
    """
    XAI step for a single network flow: decomposes its reconstruction
    error into per-feature contributions, same reconstruction-error
    attribution idea as fraud_inference.py's explain_transaction() and
    lstm_inference.py's _feature_attribution().

    UNLIKE the fraud model's PCA-anonymized V1-V28, these feature names
    are human-readable (sbytes, dttl, ct_srv_dst, proto_tcp, etc.) --
    same interpretability advantage the LSTM model has, worth noting in
    your XAI section as a point of contrast with the fraud model's
    documented interpretability gap.

    row: a single-row pandas Series/DataFrame row with the raw UNSW-NB15
         columns (before encoding).
    Returns: [{"feature": str, "contribution_pct": float}, ...]
    """
    _load()

    row_df = row.to_frame().T if hasattr(row, "to_frame") else pd.DataFrame([row])
    X = _encode(row_df)
    X_scaled = _scaler.transform(X)

    recon = _model.predict(X_scaled)
    sq_err = np.square(X_scaled - recon)[0]
    total = float(sq_err.sum()) or 1e-9

    order = np.argsort(-sq_err)[:top_n]
    return [
        {"feature": _encoder_columns[i], "contribution_pct": round(float(sq_err[i] / total * 100), 1)}
        for i in order
    ]


def score_flows(df: pd.DataFrame):
    """
    df must contain the raw UNSW-NB15 feature columns (proto, service,
    state, dur, sbytes, dbytes, ... -- everything except id/attack_cat/label).
    Returns a DataFrame with added columns: reconstruction_error,
    is_attack, intrusion_score (0-100, same scale convention as fraud_score).
    """
    _load()

    X = _encode(df)
    X_scaled = _scaler.transform(X)

    reconstructions = _model.predict(X_scaled)
    errors = np.mean(np.square(X_scaled - reconstructions), axis=1)

    result = df.copy()
    result["reconstruction_error"] = errors
    result["is_attack"] = errors > _threshold
    result["intrusion_score"] = np.clip((errors / _threshold) * 50, 0, 100).round(1)

    return result


if __name__ == "__main__":
    # Quick smoke test using a slice of the official test set
    sample = pd.read_csv("dataset/cert/unsw_nb15_testing.csv").sample(10, random_state=1)
    sample_features = sample.drop(columns=["id", "attack_cat", "label"])
    scored = score_flows(sample_features)
    scored["true_label"] = sample["label"].values
    print(scored[["true_label", "reconstruction_error", "is_attack", "intrusion_score"]])

    top_row_idx = scored["intrusion_score"].idxmax()
    print(f"\nTop contributing features (XAI) for the highest-scored flow "
          f"(intrusion_score={scored.loc[top_row_idx, 'intrusion_score']}):")
    for f in explain_flow(sample_features.loc[top_row_idx]):
        print(f"  {f['feature']}: {f['contribution_pct']}%")
