"""
Validates the fast reconstruction-error feature attribution used live in
lstm_inference.py (_feature_attribution) against real SHAP values,
computed offline with shap.GradientExplainer.

WHY THIS EXISTS: same reasoning as shap_evaluate_fraud.py -- the live
LSTM attribution is fast and human-readable but isn't SHAP. This script
runs actual SHAP on a sample of behavioral sequences and reports how
closely the two methods' feature rankings agree, over the 8 named
features (hour_of_day, is_external_recipient, etc.) rather than raw
PCA components -- so, unlike the fraud model, this comparison is
directly interpretable in business terms.

RUN THIS ON YOUR OWN MACHINE, after train_lstm_local.py has already
produced lstm_autoencoder.pt, lstm_scaler.pkl, and lstm_threshold.json.

Setup:
    pip install shap scipy

Usage:
    python shap_evaluate_lstm.py

Output (saved in "evaluation_output"):
    - shap_lstm_feature_importance.png
    - shap_lstm_summary.json

NOTE ON RUNTIME: GradientExplainer is much faster than KernelExplainer
since it uses the model's actual gradients (the LSTM is a PyTorch model,
unlike the fraud Autoencoder), so this should run in well under a minute
even with everything left at the defaults below.
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from evaluate_lstm_model import load_everything
from train_lstm_local import build_sequences
import lstm_inference

OUT_DIR = "evaluation_output"
BACKGROUND_SIZE = 50
EXPLAIN_SIZE = 40
RANDOM_STATE = 42


class _ReconErrorWrapper(nn.Module):
    """Wraps the LSTM Autoencoder so its output is a single scalar per
    sequence (mean squared reconstruction error), which is the quantity
    SHAP needs to attribute back to the 8 input features -- matching what
    the anomaly decision is actually based on, not the raw reconstruction."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        recon = self.model(x)
        err = torch.mean((x - recon) ** 2, dim=(1, 2))
        return err.unsqueeze(-1)  # (batch, 1) -- GradientExplainer expects an output dim


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.RandomState(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)

    print("Loading model, scaler, threshold, and dataset...")
    df, model, scaler, meta = load_everything()
    feature_names = meta["feature_names"]

    print("Rebuilding sequences...")
    X, seq_meta = build_sequences(df)
    N, T, F = X.shape
    X_scaled = scaler.transform(X.reshape(-1, F)).reshape(N, T, F)

    idx = rng.permutation(N)
    background_idx = idx[:BACKGROUND_SIZE]
    explain_idx = idx[BACKGROUND_SIZE:BACKGROUND_SIZE + EXPLAIN_SIZE]

    background = torch.tensor(X_scaled[background_idx], dtype=torch.float32)
    X_explain = torch.tensor(X_scaled[explain_idx], dtype=torch.float32)

    print(f"Background: {len(background)} sequences")
    print(f"Explaining: {len(X_explain)} sequences")

    wrapped_model = _ReconErrorWrapper(model)
    wrapped_model.eval()

    print("\nRunning SHAP GradientExplainer...")
    explainer = shap.GradientExplainer(wrapped_model, background)
    shap_values = explainer.shap_values(X_explain)

    # shap_values shape varies slightly by shap version -- normalize to a
    # plain (batch, seq_len, feature_dim) array regardless.
    sv = shap_values[0] if isinstance(shap_values, list) else shap_values
    sv = np.array(sv)
    if sv.ndim == 4:
        sv = sv[..., 0]

    # Average absolute SHAP value over the batch and the time dimension --
    # collapses to one importance value per named feature, same shape as
    # the fast method's per-feature attribution.
    mean_abs_shap = np.abs(sv).mean(axis=(0, 1))
    shap_pct = mean_abs_shap / mean_abs_shap.sum() * 100
    shap_ranking = sorted(zip(feature_names, shap_pct), key=lambda x: -x[1])
    top_shap = {f_name: round(float(v), 2) for f_name, v in shap_ranking}

    print("\nSHAP feature importance (mean |value|, %):")
    for f_name, pct in shap_ranking:
        print(f"  {f_name}: {pct:.2f}%")

    # ---- Bar chart (a beeswarm doesn't read well with only 8 features
    # averaged over time, so a ranked bar chart communicates this more
    # clearly for the report) ----
    plt.figure(figsize=(8, 5))
    names_sorted = [f_name for f_name, _ in shap_ranking]
    vals_sorted = [v for _, v in shap_ranking]
    plt.barh(names_sorted[::-1], vals_sorted[::-1], color="#2DD4BF")
    plt.xlabel("Mean |SHAP value| (% of total)")
    plt.title("SHAP Feature Importance — LSTM Behavioral Autoencoder\n"
               "(averaged over sampled sequences and time steps)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "shap_lstm_feature_importance.png"), dpi=150)
    plt.close()
    print(f"\nSaved {OUT_DIR}/shap_lstm_feature_importance.png")

    # ---- Compare against the fast method's rankings on the SAME
    # sequences, computed via the real, live _feature_attribution() logic
    # (imported straight from lstm_inference.py, not reimplemented here) ----
    # _feature_attribution() reads lstm_inference's module-level _meta,
    # which is normally populated as a side effect of score_user_sequence().
    # We're calling _feature_attribution() directly instead, so trigger
    # that loading explicitly first -- otherwise _meta is still None.
    #
    # FAST_AGGREGATION controls which version of the fast method is being
    # validated: "max" is the corrected version (see lstm_inference.py's
    # _feature_attribution docstring) that fixed the is_large_attachment
    # underweighting this script originally surfaced. Set to "mean" to
    # reproduce the original pre-fix correlation for comparison.
    FAST_AGGREGATION = "max"

    lstm_inference._load()

    # --- DIAGNOSTIC: sanity-check that _feature_attribution actually
    # supports/honors aggregation="max" as currently defined in
    # lstm_inference.py -- this catches the case where the docstring's
    # claimed fix was never actually applied to the function body, or a
    # stale .pyc/import is shadowing an older version. ---
    import inspect
    src = inspect.getsource(lstm_inference._feature_attribution)
    print("\n--- _feature_attribution source check ---")
    print(f"'aggregation == \"max\"' present in source: {'aggregation == \"max\"' in src}")
    print(f"'.max(dim=1)' present in source: {'.max(dim=1)' in src}")
    print("------------------------------------------\n")

    # --- DIAGNOSTIC: raw (non-normalized, pre-percentage) per-feature
    # squared-error magnitude, plus how often is_large_attachment is
    # actually 1 in the explained sample -- tells us whether the signal
    # exists in the scaled feature space at all before % normalization
    # potentially washes it out. ---
    is_large_idx = feature_names.index("is_large_attachment")
    raw_max_sq_err = {f_name: [] for f_name in feature_names}
    hot_count = 0

    fast_scores = {f_name: 0.0 for f_name in feature_names}
    with torch.no_grad():
        for i in range(len(X_explain)):
            x_i = X_explain[i:i + 1]
            recon_i = model(x_i)

            sq_err = (x_i - recon_i) ** 2  # (1, T, F), SCALED feature space
            per_feature_raw = sq_err.max(dim=1).values.squeeze(0).cpu().numpy()
            for j, f_name in enumerate(feature_names):
                raw_max_sq_err[f_name].append(float(per_feature_raw[j]))

            # was this sequence's is_large_attachment ever "hot" (raw==1)
            # at any timestep? Scaled values aren't 0/1 anymore, so check
            # the unscaled sequence instead.
            raw_seq = X[explain_idx[i]]  # unscaled (T, F)
            if (raw_seq[:, is_large_idx] >= 1.0).any():
                hot_count += 1

            attrs = lstm_inference._feature_attribution(
                x_i, recon_i, top_n=len(feature_names), aggregation=FAST_AGGREGATION
            )
            for a in attrs:
                fast_scores[a["feature"]] += a["contribution_pct"]

    total = sum(fast_scores.values()) or 1e-9
    fast_pct = {f_name: v / total * 100 for f_name, v in fast_scores.items()}

    print("--- Raw max squared-error per feature (scaled space, mean over explained sample) ---")
    for f_name in feature_names:
        vals = raw_max_sq_err[f_name]
        marker = "  <-- is_large_attachment" if f_name == "is_large_attachment" else ""
        print(f"  {f_name:28s} mean={np.mean(vals):.5f}  max={np.max(vals):.5f}{marker}")
    print(f"\n  is_large_attachment was 'hot' (>=1 at some timestep, raw/unscaled) "
          f"in {hot_count}/{len(X_explain)} explained sequences.")
    if hot_count == 0:
        print("  ==> is_large_attachment never fires in this sample. Its low fast-method\n"
              "      score is expected/correct here, not a bug -- re-run with a larger\n"
              "      EXPLAIN_SIZE or a different random seed to get sequences where it's hot.")
    print("------------------------------------------------------------------------------------\n")

    fast_vals = [fast_pct[f_name] for f_name in feature_names]
    shap_vals = [top_shap[f_name] for f_name in feature_names]
    rho, pval = spearmanr(fast_vals, shap_vals)
    print(f"\nSpearman correlation between fast attribution and SHAP rankings: "
          f"rho={rho:.3f} (p={pval:.4f}, n={len(feature_names)} features)")

    summary = {
        "background_size": len(background),
        "explained_size": len(X_explain),
        "fast_method_aggregation": FAST_AGGREGATION,
        "shap_feature_importance_pct": top_shap,
        "fast_method_feature_importance_pct": {f_name: round(v, 2) for f_name, v in fast_pct.items()},
        "fast_vs_shap_correlation": {
            "spearman_rho": round(float(rho), 3),
            "p_value": round(float(pval), 4),
            "n_features_compared": len(feature_names),
        },
        "note": (
            "SHAP explains reconstruction error via shap.GradientExplainer "
            "on the PyTorch LSTM Autoencoder. Averaged over sampled "
            "sequences and time steps to produce one importance value per "
            "named feature. Validates the live/fast reconstruction-error "
            "attribution used in lstm_inference.py's _feature_attribution() "
            "-- see fast_vs_shap_correlation for how closely their feature "
            "rankings agree."
        ),
    }
    with open(os.path.join(OUT_DIR, "shap_lstm_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\nSaved {OUT_DIR}/shap_lstm_summary.json")
    print("Use the bar chart + correlation figure in your XAI validation subsection.")


if __name__ == "__main__":
    main()