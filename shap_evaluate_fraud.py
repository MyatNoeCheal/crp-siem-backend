"""
Validates the fast reconstruction-error feature attribution used live in
fraud-inference.py against real SHAP values, computed offline.

WHY THIS EXISTS (see the Evaluation Framework / XAI section of the report):
Live scoring uses a fast reconstruction-error decomposition (see
explain_transaction() in fraud-inference.py) because it's cheap enough to
run on every request. That is NOT SHAP, even though it's inspired by the
same idea. This script runs actual SHAP (shap.KernelExplainer -- model-
agnostic, works with the sklearn MLPRegressor Autoencoder) on a sample of
transactions, then reports how closely the fast method's feature ranking
agrees with SHAP's -- giving you a legitimate methodological validation
to cite, instead of just a caveat that the live XAI isn't "real" SHAP.

RUN THIS ON YOUR OWN MACHINE, after train_autoencoder_local.py and
evaluate_model.py have already been run (needs fraud_autoencoder.pkl,
fraud_scaler.pkl, fraud_threshold.json, dataset/cert/creditcard.csv, and
ideally evaluation_output/metrics_summary.json from evaluate_model.py so
this script can compute the fast-vs-SHAP correlation).

Setup:
    pip install shap scipy

    (numpy, pandas, scikit-learn, joblib, matplotlib should already be
    installed from earlier steps.)

Usage:
    python shap_evaluate_fraud.py

Output (saved in "evaluation_output", alongside evaluate_model.py's files):
    - shap_fraud_beeswarm.png     (SHAP summary plot)
    - shap_fraud_summary.json     (top SHAP features + correlation with
                                    the fast method's rankings)

NOTE ON RUNTIME: KernelExplainer is model-agnostic but slow -- it re-runs
the model many times per explained sample. This script deliberately
explains a SMALL sample (default 40 transactions, weighted toward the
true-positive fraud cases) rather than the full test set. That's enough
for a report-ready summary plot and a correlation check; it is not meant
to be run on thousands of rows. Expect this to take a few minutes.
"""

import os
import json
import numpy as np
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from evaluate_model import load_everything, rebuild_test_split, reconstruction_error

OUT_DIR = "evaluation_output"
BACKGROUND_SIZE = 60      # background sample for KernelExplainer
EXPLAIN_SIZE = 40         # number of transactions to actually explain
RANDOM_STATE = 42


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.RandomState(RANDOM_STATE)

    print("Loading model, scaler, threshold, and dataset...")
    df, model, scaler, meta = load_everything()
    feature_cols = meta["feature_cols"]

    print("Rebuilding the test split used during training/evaluation...")
    X_test, y_test = rebuild_test_split(df, scaler, feature_cols)

    # Background: a random sample of NORMAL transactions (what the
    # Autoencoder was trained to reconstruct well) -- standard SHAP
    # practice, keeps the background small and representative.
    normal_idx = np.where(y_test == 0)[0]
    background_idx = rng.choice(normal_idx, size=min(BACKGROUND_SIZE, len(normal_idx)), replace=False)
    background = X_test[background_idx]

    # Sample to explain: bias toward fraud cases (the ones an analyst
    # actually cares about understanding), fill the rest with normals.
    fraud_idx = np.where(y_test == 1)[0]
    n_fraud = min(len(fraud_idx), EXPLAIN_SIZE // 2)
    explain_fraud_idx = rng.choice(fraud_idx, size=n_fraud, replace=False)
    n_normal = EXPLAIN_SIZE - n_fraud
    explain_normal_idx = rng.choice(normal_idx, size=n_normal, replace=False)
    explain_idx = np.concatenate([explain_fraud_idx, explain_normal_idx])
    X_explain = X_test[explain_idx]

    print(f"Background: {len(background)} normal transactions")
    print(f"Explaining: {len(X_explain)} transactions ({n_fraud} fraud, {n_normal} normal)")

    def f(X):
        """SHAP explains this function's scalar output: reconstruction
        error, exactly the quantity the anomaly decision is based on."""
        return reconstruction_error(model, X)

    print("\nRunning SHAP KernelExplainer (this is the slow part -- "
          "expect a few minutes)...")
    explainer = shap.KernelExplainer(f, background)
    shap_values = explainer.shap_values(X_explain, nsamples="auto")
    shap_values = np.array(shap_values)

    # ---- Beeswarm summary plot ----
    plt.figure()
    shap.summary_plot(shap_values, X_explain, feature_names=feature_cols, show=False)
    plt.title("SHAP Summary — Fraud Autoencoder Reconstruction Error")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "shap_fraud_beeswarm.png"), dpi=150)
    plt.close()
    print(f"Saved {OUT_DIR}/shap_fraud_beeswarm.png")

    # ---- Top features by mean |SHAP value| ----
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_pct = mean_abs_shap / mean_abs_shap.sum() * 100
    shap_ranking = sorted(zip(feature_cols, shap_pct), key=lambda x: -x[1])
    top_shap = {f_name: round(float(v), 2) for f_name, v in shap_ranking[:15]}

    print("\nTop features by SHAP (mean |value|, %):")
    for f_name, pct in list(top_shap.items())[:5]:
        print(f"  {f_name}: {pct}%")

    # ---- Compare against the fast method's rankings, if available ----
    correlation = None
    fast_features_path = os.path.join(OUT_DIR, "metrics_summary.json")
    if os.path.exists(fast_features_path):
        with open(fast_features_path) as fh:
            fast_summary = json.load(fh)
        fast_top = fast_summary.get("top_feature_attribution_pct", {})
        if fast_top:
            shared = [f_name for f_name in feature_cols if f_name in fast_top]
            shap_rank_map = dict(shap_ranking)
            fast_vals = [fast_top.get(f_name, 0.0) for f_name in shared]
            shap_vals = [shap_rank_map.get(f_name, 0.0) for f_name in shared]
            rho, pval = spearmanr(fast_vals, shap_vals)
            correlation = {
                "spearman_rho": round(float(rho), 3),
                "p_value": round(float(pval), 4),
                "n_features_compared": len(shared),
            }
            print(f"\nSpearman correlation between fast attribution and SHAP rankings: "
                  f"rho={rho:.3f} (p={pval:.4f}, n={len(shared)} features)")
        else:
            print("\n(metrics_summary.json found, but no 'top_feature_attribution_pct' in it -- "
                  "run evaluate_model.py first for a full comparison.)")
    else:
        print(f"\n({fast_features_path} not found -- run evaluate_model.py first for a full "
              f"fast-vs-SHAP comparison.)")

    summary = {
        "background_size": len(background),
        "explained_size": len(X_explain),
        "explained_fraud_count": int(n_fraud),
        "explained_normal_count": int(n_normal),
        "top_shap_feature_attribution_pct": top_shap,
        "fast_vs_shap_correlation": correlation,
        "note": (
            "SHAP explains reconstruction error via shap.KernelExplainer "
            "(model-agnostic, since the Autoencoder is a scikit-learn "
            "MLPRegressor). This validates the live/fast reconstruction-"
            "error attribution used in fraud-inference.py's "
            "explain_transaction() -- see fast_vs_shap_correlation for how "
            "closely their feature rankings agree."
        ),
    }
    with open(os.path.join(OUT_DIR, "shap_fraud_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\nSaved {OUT_DIR}/shap_fraud_summary.json")
    print("Use the beeswarm plot + correlation figure in your XAI validation subsection.")


if __name__ == "__main__":
    main()