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
    - feature_attribution.png        (XAI: which features drive fraud flags)
    - metrics_summary.json           (also prints to console)

These four PNGs and the metrics table are exactly what belongs in your
report's Evaluation Framework and Explainable AI sections.
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


def plot_feature_attribution(X_test, y_test, preds, model, feature_cols, out_path, top_n=15):
    """
    Explainable-AI figure: for the transactions the Autoencoder correctly
    flagged as fraud (true positives), decomposes each one's reconstruction
    error into per-feature contributions and averages them -- showing which
    features the model relies on most when it flags fraud. Same underlying
    idea as lstm_inference.py's per-sequence attribution, applied in
    aggregate across the test set instead of to one live event.

    HONEST LIMITATION -- state this plainly in your report's Critical
    Evaluation / XAI section: V1-V28 are PCA-anonymized by the dataset
    provider, so this figure tells you *which* features the model leans on
    (e.g. "V14", "V4") but not what they represent in business terms --
    unlike the LSTM's human-readable features (see lstm_inference.py). This
    is a real, literature-documented interpretability gap for this dataset
    (Villegas-Ch et al., 2025; B.V, 2025), not a flaw in the technique.
    """
    true_positive_mask = (y_test == 1) & (preds == 1)
    tp_X = X_test[true_positive_mask]

    if len(tp_X) == 0:
        print("No true-positive fraud cases in the test set -- skipping "
              "feature attribution figure (nothing to explain).")
        return None

    reconstructions = model.predict(tp_X)
    sq_err = np.square(tp_X - reconstructions)          # (n_true_positives, n_features)
    avg_contribution = sq_err.mean(axis=0)
    avg_contribution_pct = avg_contribution / avg_contribution.sum() * 100

    order = np.argsort(-avg_contribution_pct)[:top_n]
    top_features = [feature_cols[i] for i in order]
    top_values = avg_contribution_pct[order]

    plt.figure(figsize=(8, 6))
    plt.barh(top_features[::-1], top_values[::-1], color="#2DD4BF")
    plt.xlabel("Average Contribution to Reconstruction Error (%)")
    plt.title(f"Feature Attribution — Autoencoder Fraud Flags\n"
              f"(n={len(tp_X)} correctly-flagged fraud transactions)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    return {f: round(float(v), 2) for f, v in zip(top_features, top_values)}


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

    print("Generating feature attribution (XAI) figure...")
    top_features = plot_feature_attribution(
        X_test, y_test, preds, model, feature_cols,
        os.path.join(OUT_DIR, "feature_attribution.png")
    )

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
        "top_feature_attribution_pct": top_features,
        "xai_note": (
            "V1-V28 are PCA-anonymized by the dataset provider -- this "
            "shows WHICH features drive fraud flags, not what they mean "
            "in business terms. State this limitation explicitly in the "
            "report's XAI discussion."
        ),
    }

    with open(os.path.join(OUT_DIR, "metrics_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Final Test Set Metrics (report-ready) ===")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1 Score:  {f1:.3f}")
    print(f"AUPRC:     {auprc:.3f}")
    print(f"Confusion Matrix: TN={tn} FP={fp} FN={fn} TP={tp}")
    if top_features:
        print("\nTop features driving fraud flags (XAI):")
        for f, pct in list(top_features.items())[:5]:
            print(f"  {f}: {pct}%")
    print(f"\nAll charts + metrics_summary.json saved to ./{OUT_DIR}/")
    print("Insert the four PNGs directly into your Evaluation Framework / XAI section.")


if __name__ == "__main__":
    main()