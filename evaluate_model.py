"""
Generates report-ready evaluation visuals and metrics for the fraud
Autoencoder, using the real trained model and the real creditcard.csv.

RUN THIS ON YOUR OWN MACHINE, after train_autoencoder_local.py has
already produced fraud_autoencoder.pkl, fraud_scaler.pkl, and
fraud_threshold.json in the same folder.

Setup:
    pip install numpy pandas scikit-learn matplotlib joblib

Usage:
    python evaluate_model.py

Output (all saved in an "evaluation_output" folder):
    - precision_recall_curve.png
    - confusion_matrix.png
    - reconstruction_error_distribution.png
    - metrics_summary.json   (also prints to console)

These three PNGs and the metrics table are exactly what belongs in your
report's Evaluation Framework section.
"""

import numpy as np
import pandas as pd
import json
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_recall_curve, average_precision_score,
    precision_score, recall_score, f1_score, confusion_matrix
)
import matplotlib
matplotlib.use("Agg")  # no GUI needed, just save files
import matplotlib.pyplot as plt

DATA_PATH = "dataset/cert/creditcard.csv"   # same path used in training
MODEL_PATH = "fraud_autoencoder.pkl"
SCALER_PATH = "fraud_scaler.pkl"
THRESHOLD_PATH = "fraud_threshold.json"
OUT_DIR = "evaluation_output"

RANDOM_STATE = 42  # must match train_autoencoder_local.py so the test split lines up


def load_everything():
    for path in [DATA_PATH, MODEL_PATH, SCALER_PATH, THRESHOLD_PATH]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Couldn't find '{path}'. Make sure you've already run "
                f"train_autoencoder_local.py and that this script is in "
                f"the same folder as its output files."
            )

    df = pd.read_csv(DATA_PATH)
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    with open(THRESHOLD_PATH) as f:
        meta = json.load(f)

    return df, model, scaler, meta


def rebuild_test_split(df, scaler, feature_cols):
    """Reproduces the exact same train/val/test split used during training,
    so the metrics here match what training reported."""
    X = df[feature_cols].copy()
    y = df["Class"].values
    X[["Time", "Amount"]] = scaler.transform(X[["Time", "Amount"]])
    X = X.values

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_test, y_test, test_size=0.5, random_state=RANDOM_STATE, stratify=y_test
    )
    return X_test, y_test


def reconstruction_error(model, X):
    reconstructions = model.predict(X)
    return np.mean(np.square(X - reconstructions), axis=1)


def plot_precision_recall(y_test, errors, threshold, out_path):
    precisions, recalls, thresholds = precision_recall_curve(y_test, errors)
    auprc = average_precision_score(y_test, errors)

    plt.figure(figsize=(7, 5))
    plt.plot(recalls, precisions, color="#2DD4BF", linewidth=2, label=f"AUPRC = {auprc:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve — Fraud Autoencoder")
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return auprc


def plot_confusion_matrix(y_test, preds, out_path):
    cm = confusion_matrix(y_test, preds)
    labels = ["Normal", "Fraud"]

    plt.figure(figsize=(5.5, 5))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix — Test Set")
    plt.colorbar()
    plt.xticks([0, 1], labels)
    plt.yticks([0, 1], labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=14)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return cm


def plot_error_distribution(y_test, errors, threshold, out_path):
    normal_errors = errors[y_test == 0]
    fraud_errors = errors[y_test == 1]

    plt.figure(figsize=(7, 5))
    plt.hist(normal_errors, bins=50, alpha=0.6, label="Normal", color="#2DD4BF", density=True)
    plt.hist(fraud_errors, bins=50, alpha=0.6, label="Fraud", color="#F0465B", density=True)
    plt.axvline(threshold, color="#EAC54F", linestyle="--", linewidth=2, label=f"Threshold = {threshold:.4f}")
    plt.xlabel("Reconstruction Error")
    plt.ylabel("Density")
    plt.title("Reconstruction Error Distribution — Normal vs Fraud")
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
    feature_cols = meta["feature_cols"]

    print("Rebuilding the exact test split used during training...")
    X_test, y_test = rebuild_test_split(df, scaler, feature_cols)
    print(f"Test set: {len(X_test)} transactions ({(y_test == 1).sum()} fraud)")

    print("Scoring test set...")
    errors = reconstruction_error(model, X_test)
    preds = (errors > threshold).astype(int)

    precision = precision_score(y_test, preds, zero_division=0)
    recall = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)

    print("\nGenerating precision-recall curve...")
    auprc = plot_precision_recall(y_test, errors, threshold,
                                    os.path.join(OUT_DIR, "precision_recall_curve.png"))

    print("Generating confusion matrix...")
    cm = plot_confusion_matrix(y_test, preds,
                                 os.path.join(OUT_DIR, "confusion_matrix.png"))

    print("Generating reconstruction error distribution...")
    plot_error_distribution(y_test, errors, threshold,
                              os.path.join(OUT_DIR, "reconstruction_error_distribution.png"))

    tn, fp, fn, tp = cm.ravel()
    summary = {
        "test_set_size": int(len(X_test)),
        "test_set_fraud_count": int((y_test == 1).sum()),
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "auprc": float(auprc),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
    }

    with open(os.path.join(OUT_DIR, "metrics_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Final Test Set Metrics (report-ready) ===")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1 Score:  {f1:.3f}")
    print(f"AUPRC:     {auprc:.3f}")
    print(f"Confusion Matrix: TN={tn} FP={fp} FN={fn} TP={tp}")
    print(f"\nAll charts + metrics_summary.json saved to ./{OUT_DIR}/")
    print("Insert the three PNGs directly into your Evaluation Framework section.")


if __name__ == "__main__":
    main()