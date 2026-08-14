"""
Simulated live network-intrusion stream for the dashboard demo.

WHY THIS EXISTS: same reasoning as fraud_stream.py. Real live SIEM events
(logins, payments, admin actions) don't naturally produce UNSW-NB15's
45 flow-level features (packet/byte counts, TCP timing, protocol/service/
state, etc.) -- those come from a network flow capture tool (Argus/
Bro-IDS), not your application layer. This module replays held-out
UNSW-NB15 test-set rows through the real trained Network Intrusion
Autoencoder, clearly labelled as a simulated stream.

Inserted with category="threat" (not a new dashboard tab) since network
intrusion is conceptually a threat-detection signal -- it flows straight
into the existing Threats tab / alert pipeline alongside the rule-based
failed-login and port-scan detections.

Requires (same as network_inference.py):
    dataset/cert/unsw_nb15_testing.csv
    network_autoencoder.pkl, network_scaler.pkl,
    network_encoder_columns.pkl, network_threshold.json
"""

import os
import random
from datetime import datetime, timedelta

import pandas as pd

from network_inference import score_flows, explain_flow

DATA_PATH = "dataset/cert/unsw_nb15_testing.csv"
DROP_COLUMNS = ["id", "attack_cat", "label"]

# Small fixed pool of synthetic source IPs -- UNSW-NB15's official
# partition doesn't include real IP addresses (stripped for privacy), so
# these are purely for demo readability on the dashboard.
_DEMO_IPS = [f"203.0.113.{n}" for n in range(10, 90)]


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
            f"Couldn't find '{DATA_PATH}'. The simulated network stream "
            f"needs the official UNSW-NB15 test partition."
        )
    df = pd.read_csv(DATA_PATH)
    return df.sample(n=min(sample_size, len(df)), random_state=random_state)


def generate_simulated_network_events(count=20, random_state=None):
    """
    Scores `count` replayed network flows with the real trained Network
    Intrusion Autoencoder and returns them as LogEvent-compatible dicts,
    ready to insert into logs_collection / pass to _upsert_alert.

    Biases the sample toward including some attack flows (up to half),
    same balance approach as fraud_stream.py's
    generate_simulated_fraud_events() -- keeps the demo from being 20
    identical "normal" rows without cherry-picking only attacks.
    """
    pool = _load_pool(sample_size=max(count * 5, 200), random_state=random_state)
    attack_cats = pool["attack_cat"].copy()
    features = pool.drop(columns=DROP_COLUMNS)

    scored = score_flows(features)
    scored["attack_cat"] = attack_cats.values

    attack_rows = scored[scored["is_attack"]]
    normal_rows = scored[~scored["is_attack"]]

    n_attack = min(len(attack_rows), count // 2)
    n_normal = min(count - n_attack, len(normal_rows))

    parts = []
    if n_attack:
        parts.append(attack_rows.sample(n=n_attack, random_state=random_state))
    if n_normal:
        parts.append(normal_rows.sample(n=n_normal, random_state=random_state))
    chosen = pd.concat(parts) if parts else scored.head(0)

    now = datetime.utcnow()
    events = []
    for i, (_, row) in enumerate(chosen.iterrows()):
        is_anomaly = bool(row["is_attack"])
        intrusion_score = float(row["intrusion_score"])

        feature_row = features.loc[row.name]
        top_features = explain_flow(feature_row, top_n=3)

        attack_cat = row.get("attack_cat", "Normal")
        reason = []
        if is_anomaly:
            reason.append(
                f"Network Autoencoder reconstruction error "
                f"{row['reconstruction_error']:.4f} above threshold "
                f"(dataset label: {attack_cat})"
            )
        reason += [
            f"{f['feature']} ({f['contribution_pct']}% contribution)"
            for f in top_features
        ]

        events.append({
            "ip": random.choice(_DEMO_IPS),
            "event_type": "network_intrusion" if is_anomaly else "network_flow",
            "severity": "high" if is_anomaly else "low",
            "timestamp": (now - timedelta(seconds=i * 5)).isoformat(),
            "role": "user",
            "category": "threat",
            "source": "simulated_network_stream",
            "anomaly": is_anomaly,
            "risk_score": intrusion_score,
            "risk_level": _risk_level(intrusion_score),
            "reason": reason,
            "top_features": top_features,
            "reconstruction_error": float(row["reconstruction_error"]),
            "attack_cat": attack_cat,
        })

    return events
