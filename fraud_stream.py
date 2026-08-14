"""
Simulated live fraud-transaction stream for the dashboard demo.

WHY THIS EXISTS: fraud_inference.py's own docstring lays out two honest
ways to use the trained Autoencoder. Real e-commerce checkout events don't
naturally produce the V1-V28 PCA-anonymized features the model was trained
on (see fraud_inference.py's module docstring), so this module implements
Option 2 from that docstring: replay held-out transactions from the
training dataset through the live pipeline, clearly labelled as a
simulated stream -- not silently pretending real checkout traffic is
being scored.

Each simulated event is shaped exactly like a real /detect LogEvent dict,
so it flows through the same logs_collection insert + _upsert_alert path
as everything else. The only difference is `source: "simulated_fraud_stream"`
and the extra fraud_score / reconstruction_error / XAI fields, which the
Fraud tab can surface directly.

Requires (same as fraud_inference.py):
    dataset/cert/creditcard.csv
    fraud_autoencoder.pkl, fraud_scaler.pkl, fraud_threshold.json
    (all already present if you've run train_autoencoder_local.py)
"""

import os
import random
from datetime import datetime, timedelta

import pandas as pd

from fraud_inference import score_transactions, explain_transaction

DATA_PATH = "dataset/cert/creditcard.csv"

# Small fixed pool of synthetic customer IDs / IPs so the Fraud tab reads
# like varied traffic instead of one repeated identity. Not meant to
# represent real customers -- purely for demo readability.
_DEMO_USERS = [f"cust_{n}" for n in range(9000, 9020)]
_DEMO_IPS = [f"192.0.2.{n}" for n in range(10, 90)]


def _risk_level(score):
    if score >= 85:
        return "Critical"
    elif score >= 60:
        return "High"
    elif score >= 30:
        return "Medium"
    return "Low"


def _load_pool(sample_size, random_state=None):
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Couldn't find '{DATA_PATH}'. The simulated fraud stream needs "
            f"the same creditcard.csv used to train the Autoencoder."
        )
    df = pd.read_csv(DATA_PATH)
    return df.sample(n=min(sample_size, len(df)), random_state=random_state)


def generate_simulated_fraud_events(count=20, random_state=None):
    """
    Scores `count` replayed transactions with the real trained Autoencoder
    and returns them as a list of dicts ready to insert into
    logs_collection / pass to _upsert_alert, exactly like any other event.

    Biases the sample toward including some of the model's own flagged
    fraud cases (up to half), so the demo isn't 20 identical "normal"
    rows -- the rest are a random draw, so it still looks like real mixed
    traffic rather than a cherry-picked fraud showcase.
    """
    pool = _load_pool(sample_size=max(count * 5, 200), random_state=random_state)
    scored = score_transactions(pool)

    fraud_rows = scored[scored["is_fraud"]]
    normal_rows = scored[~scored["is_fraud"]]

    n_fraud = min(len(fraud_rows), count // 2)
    n_normal = min(count - n_fraud, len(normal_rows))

    parts = []
    if n_fraud:
        parts.append(fraud_rows.sample(n=n_fraud, random_state=random_state))
    if n_normal:
        parts.append(normal_rows.sample(n=n_normal, random_state=random_state))
    chosen = pd.concat(parts) if parts else scored.head(0)

    now = datetime.utcnow()
    events = []
    for i, (_, row) in enumerate(chosen.iterrows()):
        is_anomaly = bool(row["is_fraud"])
        fraud_score = float(row["fraud_score"])

        top_features = explain_transaction(row, top_n=3)
        reason = []
        if is_anomaly:
            reason.append(
                f"Autoencoder reconstruction error {row['reconstruction_error']:.3f} "
                f"above threshold"
            )
        reason += [
            f"{f['feature']} ({f['contribution_pct']}% contribution)"
            for f in top_features
        ]

        events.append({
            "ip": random.choice(_DEMO_IPS),
            "user_id": random.choice(_DEMO_USERS),
            "event_type": "payment",
            "severity": "high" if is_anomaly else "low",
            "timestamp": (now - timedelta(seconds=i * 7)).isoformat(),
            "role": "user",
            "category": "fraud",
            "amount": float(row["Amount"]),
            "source": "simulated_fraud_stream",
            "anomaly": is_anomaly,
            "risk_score": fraud_score,
            "risk_level": _risk_level(fraud_score),
            "reason": reason,
            "top_features": top_features,
            "reconstruction_error": float(row["reconstruction_error"]),
        })

    return events