"""
Generates report-ready evaluation visuals for the LSTM Autoencoder,
using the real trained model and the real email.csv.

RUN THIS ON YOUR OWN MACHINE, after train_lstm_local.py has already
produced lstm_autoencoder.pt, lstm_scaler.pkl, and lstm_threshold.json.

UNSUPERVISED CAVEAT (state this plainly in your Evaluation Framework
section): email.csv has no ground-truth insider-threat labels, so there
is no Precision/Recall/F1/AUPRC to report here -- that would require
labeled fraud/insider events, which is what creditcard.csv gave the
Autoencoder and email.csv does not give the LSTM. What this script
DOES give you, and what is legitimate to report:
  - the reconstruction error distribution (shows the model separates
    "typical" from "atypical" behavior at all -- if it's one flat blob,
    the model isn't learning anything useful)
  - the specific top-N most anomalous user sequences, which you can
    describe qualitatively in the report ("sequence X involved an
    unusually large attachment sent to an external domain outside
    business hours")

Setup:
    pip install torch pandas numpy joblib matplotlib

Usage:
    python evaluate_lstm_model.py

Output (saved in an "evaluation_output" folder):
    - lstm_reconstruction_error_distribution.png
    - lstm_top_anomalies.csv
    - lstm_metrics_summary.json
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from train_lstm_local import (
    build_sequences, LSTMAutoencoder, reconstruction_error, DATA_PATH
)

MODEL_PATH = "lstm_autoencoder.pt"
SCALER_PATH = "lstm_scaler.pkl"
THRESHOLD_PATH = "lstm_threshold.json"
OUT_DIR = "evaluation_output"
TOP_N = 20


def load_everything():
    for path in [DATA_PATH, MODEL_PATH, SCALER_PATH, THRESHOLD_PATH]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Couldn't find '{path}'. Run train_lstm_local.py first."
            )

    with open(THRESHOLD_PATH) as f:
        meta = json.load(f)

    model = LSTMAutoencoder(
        feature_dim=meta["feature_dim"],
        hidden_dim=meta["hidden_dim"],
        bottleneck_dim=meta["bottleneck_dim"],
        seq_len=meta["seq_len"],
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    scaler = joblib.load(SCALER_PATH)

    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])

    return df, model, scaler, meta


def plot_error_distribution(errors, threshold, out_path):
    plt.figure(figsize=(7, 5))
    plt.hist(errors, bins=60, color="#2DD4BF", alpha=0.75)
    plt.axvline(threshold, color="#F0465B", linestyle="--", linewidth=2,
                label=f"Threshold = {threshold:.4f} ({(errors > threshold).mean()*100:.1f}% flagged)")
    plt.xlabel("Reconstruction Error (MSE)")
    plt.ylabel("Number of Sequences")
    plt.title("LSTM Autoencoder — Reconstruction Error Distribution\n(behavioral sequences, email.csv)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading model, scaler, threshold, and dataset...")
    df, model, scaler, meta = load_everything()
    threshold = meta["threshold"]

    print("Rebuilding sequences (same logic as training)...")
    X, seq_meta = build_sequences(df)
    print(f"Built {len(X)} sequences")

    N, T, F = X.shape
    X_scaled = scaler.transform(X.reshape(-1, F)).reshape(N, T, F)

    print("Scoring all sequences...")
    errors = reconstruction_error(model, X_scaled, torch.device("cpu"))

    flagged = errors > threshold
    print(f"{flagged.sum()} / {len(errors)} sequences flagged as anomalous "
          f"({flagged.mean()*100:.2f}%)")

    print("Generating reconstruction error distribution chart...")
    plot_error_distribution(
        errors, threshold,
        os.path.join(OUT_DIR, "lstm_reconstruction_error_distribution.png")
    )

    # Top-N most anomalous sequences, for qualitative discussion in the report
    order = np.argsort(-errors)[:TOP_N]
    top_rows = []
    for idx in order:
        m = seq_meta[idx]
        top_rows.append({
            "user": m["user"],
            "window_start_idx": m["window_start_idx"],
            "reconstruction_error": float(errors[idx]),
            "is_anomaly": bool(errors[idx] > threshold),
        })
    top_df = pd.DataFrame(top_rows)
    top_df.to_csv(os.path.join(OUT_DIR, "lstm_top_anomalies.csv"), index=False)

    summary = {
        "total_sequences": int(len(errors)),
        "flagged_count": int(flagged.sum()),
        "flagged_rate": float(flagged.mean()),
        "threshold": float(threshold),
        "error_mean": float(np.mean(errors)),
        "error_std": float(np.std(errors)),
        "error_min": float(np.min(errors)),
        "error_max": float(np.max(errors)),
        "note": (
            "Unsupervised evaluation -- no ground-truth insider labels exist "
            "in email.csv. Precision/Recall/F1/AUPRC are not reportable here; "
            "see lstm_top_anomalies.csv for qualitative review of the "
            "highest-error sequences instead."
        ),
    }

    with open(os.path.join(OUT_DIR, "lstm_metrics_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== LSTM Evaluation Summary ===")
    print(f"Total sequences:  {summary['total_sequences']}")
    print(f"Flagged anomalous: {summary['flagged_count']} ({summary['flagged_rate']*100:.2f}%)")
    print(f"Error mean/std:    {summary['error_mean']:.5f} / {summary['error_std']:.5f}")
    print(f"\nSaved chart + top anomalies + summary to ./{OUT_DIR}/")
    print("Use the chart + top-anomalies table in your Evaluation Framework section,")
    print("framed as unsupervised behavioral anomaly detection.")


if __name__ == "__main__":
    main()