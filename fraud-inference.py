"""
Loads the trained fraud Autoencoder (scikit-learn MLPRegressor) and scores
transactions.

IMPORTANT LIMITATION (be upfront about this in your report):
This model was trained on Time + V1-V28 (PCA-anonymized) + Amount, matching
the Kaggle credit card fraud dataset structure. A real e-commerce checkout
event from your own site will NOT naturally produce V1-V28 -- those come
from the card processor's internal risk model, not from your application
layer.

Two honest ways to use this in your project:

  1. OFFLINE EVALUATION (recommended, defensible): Use this script to score
     held-out transactions FROM THE SAME DATASET and report
     precision/recall/F1/AUPRC in your Evaluation Framework section. This
     demonstrates the AI technique works, without claiming it's wired into
     live checkout traffic.

  2. LIVE DEMO (use with a caveat): For the dashboard demo, score a held-out
     batch of transactions from the training dataset structure (not live
     site events) and stream those results into the /fraud endpoint as a
     "simulated transaction stream" -- clearly labelled as such.

Do NOT quietly feed live `amount`-only e-commerce events into this model --
it needs the full 30-feature vector it was trained on.

Setup:
    pip install numpy pandas scikit-learn joblib

Put this file in the same folder as fraud_autoencoder.pkl, fraud_scaler.pkl,
and fraud_threshold.json (the three files train_autoencoder_local.py produces).
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
_feature_cols = None


def _load():
    global _model, _scaler, _threshold, _feature_cols
    if _model is None:
        _model = joblib.load(_DIR / "fraud_autoencoder.pkl")
        _scaler = joblib.load(_DIR / "fraud_scaler.pkl")
        with open(_DIR / "fraud_threshold.json") as f:
            meta = json.load(f)
        _threshold = meta["threshold"]
        _feature_cols = meta["feature_cols"]


def score_transactions(df: pd.DataFrame):
    """
    df must contain columns: Time, V1..V28, Amount
    Returns a DataFrame with added columns: reconstruction_error, is_fraud, fraud_score
    """
    _load()

    X = df[_feature_cols].copy()
    X[["Time", "Amount"]] = _scaler.transform(X[["Time", "Amount"]])

    reconstructions = _model.predict(X.values)
    errors = np.mean(np.square(X.values - reconstructions), axis=1)

    result = df.copy()
    result["reconstruction_error"] = errors
    result["is_fraud"] = errors > _threshold
    # Normalize error into a 0-100 "fraud score" for the dashboard, capped at 100
    result["fraud_score"] = np.clip((errors / _threshold) * 50, 0, 100).round(1)

    return result


if __name__ == "__main__":
    # Quick smoke test using a slice of your dataset
    sample = pd.read_csv("dataset/cert/creditcard.csv").sample(10, random_state=1)
    scored = score_transactions(sample)
    print(scored[["Class", "reconstruction_error", "is_fraud", "fraud_score"]])