"""
Trains an Autoencoder for reconstruction-based fraud anomaly detection
on the real Kaggle "Credit Card Fraud Detection" dataset (creditcard.csv).

Uses scikit-learn's MLPRegressor instead of TensorFlow/Keras -- TensorFlow
does not yet ship Windows wheels for very new Python releases (e.g. 3.14),
so this avoids that entirely. Architecturally this is the same idea as a
Keras autoencoder: a feedforward network with a narrow bottleneck layer,
trained so its output reconstructs its input. The reconstruction error is
used as the anomaly score exactly as before.

RUN THIS ON YOUR OWN MACHINE -- it never leaves your computer.

Setup:
    pip install numpy pandas scikit-learn joblib

Usage:
    1. Edit DATA_PATH below to point to your creditcard.csv
       (based on your folder listing, it's at dataset/cert/creditcard.csv).
    2. Run:  python train_autoencoder_local.py
    3. When it finishes, you'll have these files in the same folder:
       - fraud_autoencoder.pkl     (the trained model)
       - fraud_scaler.pkl          (feature scaler, needed at inference time)
       - fraud_threshold.json      (anomaly threshold + evaluation metrics)

Approach (standard for highly imbalanced fraud data):
  1. Train ONLY on normal (Class == 0) transactions.
  2. The Autoencoder learns to reconstruct "normal" transaction patterns well.
  3. Fraudulent transactions, being statistically different, produce a
     higher reconstruction error (MSE between input and output).
  4. A threshold on that error becomes the anomaly/fraud score cutoff,
     chosen by maximizing F1 on a validation split.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import (
    precision_recall_curve, average_precision_score,
    precision_score, recall_score, f1_score, confusion_matrix
)
import joblib
import json
import os

DATA_PATH = "dataset/cert/creditcard.csv"   # edit this if your file is elsewhere
MODEL_OUT = "fraud_autoencoder.pkl"
SCALER_OUT = "fraud_scaler.pkl"
THRESHOLD_OUT = "fraud_threshold.json"

RANDOM_STATE = 42


def load_and_preprocess():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Couldn't find '{DATA_PATH}'. Edit DATA_PATH at the top of "
            f"this file to point to your creditcard.csv."
        )

    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} transactions "
          f"({(df['Class'] == 1).sum()} fraud, "
          f"{(df['Class'] == 0).sum()} normal, "
          f"{(df['Class'] == 1).mean() * 100:.3f}% fraud ratio)")

    feature_cols = [c for c in df.columns if c != "Class"]
    X = df[feature_cols].copy()
    y = df["Class"].values

    # Only Time and Amount need scaling; V1-V28 are already PCA-normalized.
    scaler = StandardScaler()
    X[["Time", "Amount"]] = scaler.fit_transform(X[["Time", "Amount"]])

    return X.values, y, feature_cols, scaler


def build_autoencoder():
    # Architecture: input -> 20 -> 14 -> 7 (bottleneck) -> 14 -> 20 -> output
    # MLPRegressor's hidden_layer_sizes defines everything BETWEEN input and
    # output, and we train it to reproduce its own input (X, X), which is
    # exactly what an autoencoder does.
    return MLPRegressor(
        hidden_layer_sizes=(20, 14, 7, 14, 20),
        activation="relu",
        solver="adam",
        alpha=1e-5,
        batch_size=256,
        learning_rate_init=0.001,
        max_iter=200,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=8,
        random_state=RANDOM_STATE,
        verbose=True,
    )


def reconstruction_error(model, X):
    reconstructions = model.predict(X)
    mse = np.mean(np.square(X - reconstructions), axis=1)
    return mse


def find_best_threshold(y_true, errors):
    """Pick the threshold that maximizes F1 score on the validation set."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, errors)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores[:-1])  # last point has no corresponding threshold
    return thresholds[best_idx], f1_scores[best_idx]


def main():
    print("Loading and preprocessing data...")
    X, y, feature_cols, scaler = load_and_preprocess()

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )

    # Train the autoencoder ONLY on normal transactions from the training split
    X_train_normal = X_train_full[y_train_full == 0]

    # Split remaining data into validation (for threshold picking) and test (final eval)
    X_val, X_test, y_val, y_test = train_test_split(
        X_test, y_test, test_size=0.5, random_state=RANDOM_STATE, stratify=y_test
    )

    print(f"Training on {len(X_train_normal)} normal transactions")
    print(f"Validation set: {len(X_val)} ({(y_val == 1).sum()} fraud)")
    print(f"Test set: {len(X_test)} ({(y_test == 1).sum()} fraud)")

    print("\nTraining autoencoder (this may take a few minutes on the full dataset)...")
    autoencoder = build_autoencoder()
    autoencoder.fit(X_train_normal, X_train_normal)
    print(f"Training complete. Iterations run: {autoencoder.n_iter_}")

    # Threshold selection on validation set
    val_errors = reconstruction_error(autoencoder, X_val)
    threshold, best_f1 = find_best_threshold(y_val, val_errors)
    print(f"\nSelected threshold: {threshold:.6f} (validation F1: {best_f1:.3f})")

    # Final evaluation on held-out test set
    test_errors = reconstruction_error(autoencoder, X_test)
    test_preds = (test_errors > threshold).astype(int)

    precision = precision_score(y_test, test_preds, zero_division=0)
    recall = recall_score(y_test, test_preds, zero_division=0)
    f1 = f1_score(y_test, test_preds, zero_division=0)
    auprc = average_precision_score(y_test, test_errors)
    tn, fp, fn, tp = confusion_matrix(y_test, test_preds).ravel()

    print("\n=== Test Set Results ===")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1:        {f1:.3f}")
    print(f"AUPRC:     {auprc:.3f}")
    print(f"Confusion matrix: TN={tn} FP={fp} FN={fn} TP={tp}")
    print("\n(These are the numbers to report in your Evaluation Framework section.)")

    # Save model, scaler, threshold, and feature order for deployment
    joblib.dump(autoencoder, MODEL_OUT)
    joblib.dump(scaler, SCALER_OUT)
    with open(THRESHOLD_OUT, "w") as f:
        json.dump({
            "threshold": float(threshold),
            "feature_cols": feature_cols,
            "test_metrics": {
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "auprc": float(auprc),
            }
        }, f, indent=2)

    print(f"\nSaved model to {MODEL_OUT}")
    print(f"Saved scaler to {SCALER_OUT}")
    print(f"Saved threshold + metrics to {THRESHOLD_OUT}")
    print("\nDone. These three files are what fraud_inference.py needs to load later.")


if __name__ == "__main__":
    main()