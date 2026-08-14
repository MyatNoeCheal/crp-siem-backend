"""
Trains an Autoencoder for network-intrusion anomaly detection on the
UNSW-NB15 dataset (official train/test partition) -- the third detection
capability referenced in your Research Proposal's literature review
(alongside the Fraud Autoencoder and LSTM behavioral model).

METHOD: same reconstruction-based approach as train_autoencoder_local.py
(the Fraud Autoencoder) -- train only on NORMAL traffic, then flag high
reconstruction-error records as anomalous. Kept architecturally
consistent with the fraud model deliberately: same technique proven to
work on creditcard.csv, same MLPRegressor substitute for the Keras
Autoencoder your proposal originally specified (see that script's
docstring for why -- no Windows Python 3.14 TensorFlow wheels).

UNLIKE creditcard.csv, UNSW-NB15 ships with an official pre-split
train/test partition (UNSW_NB15_training-set.csv / testing-set.csv), so
this script doesn't need to carve its own held-out test set the way
train_autoencoder_local.py had to -- the official testing-set.csv IS the
held-out test set, and results against it are directly comparable to the
published literature (Moustafa & Slay, and others in your Lit Review).

Setup:
    pip install numpy pandas scikit-learn joblib

Get the data first:
    Download UNSW_NB15_training-set.csv and UNSW_NB15_testing-set.csv from
    https://research.unsw.edu.au/projects/unsw-nb15-dataset
    Place them at:
        dataset/cert/unsw_nb15_training.csv
        dataset/cert/unsw_nb15_testing.csv

Usage:
    python train_network_intrusion_local.py

Output (same folder):
    - network_autoencoder.pkl     (trained model)
    - network_scaler.pkl          (StandardScaler for numeric features)
    - network_encoder_columns.pkl (the one-hot column layout, needed at
                                    inference time so live/replayed rows
                                    line up with what the model expects)
    - network_threshold.json      (threshold + evaluation metrics + feature list)
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import (
    precision_recall_curve, average_precision_score,
    precision_score, recall_score, f1_score, confusion_matrix
)
import joblib

TRAIN_PATH = "dataset/cert/unsw_nb15_training.csv"
TEST_PATH = "dataset/cert/unsw_nb15_testing.csv"

MODEL_OUT = "network_autoencoder.pkl"
SCALER_OUT = "network_scaler.pkl"
COLUMNS_OUT = "network_encoder_columns.pkl"
THRESHOLD_OUT = "network_threshold.json"

RANDOM_STATE = 42

# Columns that aren't real detection features -- id is a row index,
# attack_cat is the multi-class label (label is the binary one we use),
# and label is the target itself.
DROP_COLUMNS = ["id", "attack_cat", "label"]
CATEGORICAL_COLUMNS = ["proto", "service", "state"]


def load_and_preprocess():
    for path in [TRAIN_PATH, TEST_PATH]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Couldn't find '{path}'. Download UNSW_NB15_training-set.csv "
                f"and UNSW_NB15_testing-set.csv from "
                f"https://research.unsw.edu.au/projects/unsw-nb15-dataset "
                f"and place them at the paths above (renamed to match)."
            )

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    print(f"Loaded {len(train_df)} training rows, {len(test_df)} test rows")
    print(f"Training set attack ratio: {(train_df['label'] == 1).mean() * 100:.2f}%")
    print(f"Test set attack ratio:     {(test_df['label'] == 1).mean() * 100:.2f}%")

    y_train = train_df["label"].values
    y_test = test_df["label"].values

    # One-hot encode proto/service/state. Fit the column layout on the
    # UNION of train+test so a category that only appears in one split
    # doesn't break alignment -- then re-split back apart.
    combined = pd.concat(
        [train_df.drop(columns=DROP_COLUMNS), test_df.drop(columns=DROP_COLUMNS)],
        keys=["train", "test"],
    )
    combined_encoded = pd.get_dummies(combined, columns=CATEGORICAL_COLUMNS)

    X_train = combined_encoded.loc["train"].reset_index(drop=True)
    X_test = combined_encoded.loc["test"].reset_index(drop=True)

    numeric_cols = [c for c in X_train.columns if c not in CATEGORICAL_COLUMNS]
    encoder_columns = list(X_train.columns)  # full post-encoding column layout

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=encoder_columns
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=encoder_columns
    )

    return X_train_scaled, y_train, X_test_scaled, y_test, scaler, encoder_columns


def build_autoencoder():
    # Same architecture idea as the Fraud Autoencoder: input -> bottleneck
    # -> output, trained to reconstruct its own input. Layer sizes scaled
    # up slightly since this feature space (~190 columns after one-hot
    # encoding) is wider than creditcard.csv's 30.
    return MLPRegressor(
        hidden_layer_sizes=(64, 32, 16, 32, 64),
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
    return np.mean(np.square(X - reconstructions), axis=1)


def find_best_threshold(y_true, errors):
    precisions, recalls, thresholds = precision_recall_curve(y_true, errors)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores[:-1])
    return thresholds[best_idx], f1_scores[best_idx]


def main():
    print("Loading and preprocessing UNSW-NB15...")
    X_train, y_train, X_test, y_test, scaler, encoder_columns = load_and_preprocess()

    # Train the autoencoder ONLY on normal traffic from the training split,
    # same normal-only approach as the Fraud Autoencoder.
    X_train_normal = X_train[y_train == 0].values

    # Carve a small validation slice out of the normal training rows, for
    # threshold selection (mirrors max F1 approach in train_autoencoder_local.py)
    n_val = min(10000, len(X_train_normal) // 5)
    rng = np.random.RandomState(RANDOM_STATE)
    val_idx = rng.choice(len(X_train_normal), size=n_val, replace=False)
    val_mask = np.zeros(len(X_train_normal), dtype=bool)
    val_mask[val_idx] = True

    # Validation set for threshold-picking needs BOTH classes, so pull a
    # matching amount of attack traffic from the training split too.
    X_train_attack = X_train[y_train == 1].values
    n_val_attack = min(len(X_train_attack), n_val)
    val_attack_idx = rng.choice(len(X_train_attack), size=n_val_attack, replace=False)

    X_val = np.vstack([X_train_normal[val_mask], X_train_attack[val_attack_idx]])
    y_val = np.concatenate([np.zeros(val_mask.sum()), np.ones(n_val_attack)])

    X_train_fit = X_train_normal[~val_mask]

    print(f"Training on {len(X_train_fit)} normal records")
    print(f"Validation set: {len(X_val)} ({int(y_val.sum())} attack)")
    print(f"Held-out test set (official UNSW-NB15 testing-set.csv): "
          f"{len(X_test)} ({int((y_test == 1).sum())} attack)")

    print("\nTraining network intrusion autoencoder "
          "(this may take a few minutes)...")
    autoencoder = build_autoencoder()
    autoencoder.fit(X_train_fit, X_train_fit)
    print(f"Training complete. Iterations run: {autoencoder.n_iter_}")

    val_errors = reconstruction_error(autoencoder, X_val)
    threshold, best_f1 = find_best_threshold(y_val, val_errors)
    print(f"\nSelected threshold: {threshold:.6f} (validation F1: {best_f1:.3f})")

    # Final evaluation on the OFFICIAL held-out test set -- directly
    # comparable to published UNSW-NB15 literature results.
    test_errors = reconstruction_error(autoencoder, X_test.values)
    test_preds = (test_errors > threshold).astype(int)

    precision = precision_score(y_test, test_preds, zero_division=0)
    recall = recall_score(y_test, test_preds, zero_division=0)
    f1 = f1_score(y_test, test_preds, zero_division=0)
    auprc = average_precision_score(y_test, test_errors)
    tn, fp, fn, tp = confusion_matrix(y_test, test_preds).ravel()

    print("\n=== Test Set Results (official UNSW-NB15 testing-set.csv) ===")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1:        {f1:.3f}")
    print(f"AUPRC:     {auprc:.3f}")
    print(f"Confusion matrix: TN={tn} FP={fp} FN={fn} TP={tp}")

    joblib.dump(autoencoder, MODEL_OUT)
    joblib.dump(scaler, SCALER_OUT)
    joblib.dump(encoder_columns, COLUMNS_OUT)
    with open(THRESHOLD_OUT, "w") as f:
        json.dump({
            "threshold": float(threshold),
            "encoder_columns": encoder_columns,
            "categorical_columns": CATEGORICAL_COLUMNS,
            "test_metrics": {
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "auprc": float(auprc),
            },
            "confusion_matrix": {
                "true_negative": int(tn), "false_positive": int(fp),
                "false_negative": int(fn), "true_positive": int(tp),
            },
            "note": (
                "Evaluated against the OFFICIAL UNSW-NB15 testing-set.csv "
                "partition, not a self-carved split -- directly comparable "
                "to published literature using the same partition."
            ),
        }, f, indent=2)

    print(f"\nSaved model to {MODEL_OUT}")
    print(f"Saved scaler to {SCALER_OUT}")
    print(f"Saved column layout to {COLUMNS_OUT}")
    print(f"Saved threshold + metrics to {THRESHOLD_OUT}")
    print("\nNext: run evaluate_network_intrusion_model.py for report-ready charts,")
    print("then network_inference.py is ready for live/simulated scoring.")


if __name__ == "__main__":
    main()