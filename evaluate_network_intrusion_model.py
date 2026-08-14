"""
Generates report-ready evaluation visuals and metrics for the Network
Intrusion Autoencoder, using the real trained model and the official
UNSW-NB15 testing-set.csv.

RUN THIS ON YOUR OWN MACHINE, after train_network_intrusion_local.py has
already produced network_autoencoder.pkl, network_scaler.pkl,
network_encoder_columns.pkl, and network_threshold.json.

Setup:
    pip install numpy pandas scikit-learn matplotlib joblib

Usage:
    python evaluate_network_intrusion_model.py

Output (saved in "evaluation_output", alongside the fraud model's files):
    - network_precision_recall_curve.png
    - network_confusion_matrix.png
    - network_reconstruction_error_distribution.png
    - network_feature_attribution.png   (XAI)
    - network_metrics_summary.json
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (
    precision_recall_curve, average_precision_score, confusion_matrix
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from train_network_intrusion_local import (
    TEST_PATH, DROP_COLUMNS, CATEGORICAL_COLUMNS, reconstruction_error
)

MODEL_PATH = "network_autoencoder.pkl"
SCALER_PATH = "network_scaler.pkl"
COLUMNS_PATH = "network_encoder_columns.pkl"
THRESHOLD_PATH = "network_threshold.json"
OUT_DIR = "evaluation_output"


def load_everything():
    for path in [TEST_PATH, MODEL_PATH, SCALER_PATH, COLUMNS_PATH, THRESHOLD_PATH]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Couldn't find '{path}'. Run train_network_intrusion_local.py first."
            )

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    encoder_columns = joblib.load(COLUMNS_PATH)
    with open(THRESHOLD_PATH) as f:
        meta = json.load(f)

    test_df = pd.read_csv(TEST_PATH)
    y_test = test_df["label"].values

    X_test = pd.get_dummies(
        test_df.drop(columns=DROP_COLUMNS), columns=CATEGORICAL_COLUMNS
    )
    # Align columns to training-time layout (covers any category present
    # in test but not train, or vice versa) -- fill missing with 0.
    X_test = X_test.reindex(columns=encoder_columns, fill_value=0)
    X_test_scaled = scaler.transform(X_test)

    return X_test_scaled, y_test, model, meta, encoder_columns


def plot_precision_recall(y_test, errors, out_path):
    precisions, recalls, thresholds = precision_recall_curve(y_test, errors)
    auprc = average_precision_score(y_test, errors)

    plt.figure(figsize=(7, 5))
    plt.plot(recalls, precisions, color="#2DD4BF", linewidth=2, label=f"AUPRC = {auprc:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve — Network Intrusion Autoencoder")
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return auprc


def plot_confusion_matrix(y_test, preds, out_path):
    cm = confusion_matrix(y_test, preds)
    labels = ["Normal", "Attack"]

    plt.figure(figsize=(5.5, 5))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix — Network Intrusion (UNSW-NB15 test set)")
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
    attack_errors = errors[y_test == 1]

    plt.figure(figsize=(7, 5))
    plt.hist(normal_errors, bins=50, alpha=0.6, label="Normal", color="#2DD4BF", density=True)
    plt.hist(attack_errors, bins=50, alpha=0.6, label="Attack", color="#F0465B", density=True)
    plt.axvline(threshold, color="#EAC54F", linestyle="--", linewidth=2, label=f"Threshold = {threshold:.4f}")
    plt.xlabel("Reconstruction Error")
    plt.ylabel("Density")
    plt.title("Reconstruction Error Distribution — Normal vs Attack Traffic")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_feature_attribution(X_test, y_test, preds, model, encoder_columns, out_path, top_n=15):
    """Same aggregate reconstruction-error decomposition used by the Fraud
    Autoencoder's evaluate_model.py -- averaged over correctly-flagged
    attacks (true positives) to show which of the ~190 one-hot/numeric
    features the model leans on most when it flags an attack."""
    true_positive_mask = (y_test == 1) & (preds == 1)
    tp_X = X_test[true_positive_mask]

    if len(tp_X) == 0:
        print("No true-positive attacks in the test set -- skipping feature attribution.")
        return None

    reconstructions = model.predict(tp_X)
    sq_err = np.square(tp_X - reconstructions)
    avg_contribution = sq_err.mean(axis=0)
    avg_contribution_pct = avg_contribution / avg_contribution.sum() * 100

    order = np.argsort(-avg_contribution_pct)[:top_n]
    top_features = [encoder_columns[i] for i in order]
    top_values = avg_contribution_pct[order]

    plt.figure(figsize=(8, 6))
    plt.barh(top_features[::-1], top_values[::-1], color="#2DD4BF")
    plt.xlabel("Average Contribution to Reconstruction Error (%)")
    plt.title(f"Feature Attribution — Network Intrusion Flags\n"
              f"(n={len(tp_X)} correctly-flagged attacks)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    return {f: round(float(v), 2) for f, v in zip(top_features, top_values)}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading model, scaler, threshold, and test set...")
    X_test, y_test, model, meta, encoder_columns = load_everything()
    threshold = meta["threshold"]

    print("Scoring test set...")
    errors = reconstruction_error(model, X_test)
    preds = (errors > threshold).astype(int)

    print("Generating precision-recall curve...")
    auprc = plot_precision_recall(y_test, errors,
                                    os.path.join(OUT_DIR, "network_precision_recall_curve.png"))

    print("Generating confusion matrix...")
    cm = plot_confusion_matrix(y_test, preds,
                                 os.path.join(OUT_DIR, "network_confusion_matrix.png"))

    print("Generating reconstruction error distribution...")
    plot_error_distribution(y_test, errors, threshold,
                              os.path.join(OUT_DIR, "network_reconstruction_error_distribution.png"))

    print("Generating feature attribution (XAI) figure...")
    top_features = plot_feature_attribution(
        X_test, y_test, preds, model, encoder_columns,
        os.path.join(OUT_DIR, "network_feature_attribution.png")
    )

    tn, fp, fn, tp = cm.ravel()
    summary = {
        "test_set_size": int(len(X_test)),
        "test_set_attack_count": int((y_test == 1).sum()),
        "threshold": float(threshold),
        "auprc": float(auprc),
        "confusion_matrix": {
            "true_negative": int(tn), "false_positive": int(fp),
            "false_negative": int(fn), "true_positive": int(tp),
        },
        "top_feature_attribution_pct": top_features,
        "note": (
            "Evaluated against the OFFICIAL UNSW-NB15 testing-set.csv "
            "partition -- directly comparable to published literature "
            "using the same benchmark split."
        ),
    }
    with open(os.path.join(OUT_DIR, "network_metrics_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nAll charts + metrics saved to ./{OUT_DIR}/")
    print("Use these alongside the Fraud Autoencoder's charts in your "
          "Evaluation Framework section.")


if __name__ == "__main__":
    main()