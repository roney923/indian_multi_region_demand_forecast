"""
===========================================================
SHAP + Feature Importance for LSTM Models
Indian Multi-Region Electricity Demand Forecasting

Mirrors what shap_analysis.py does for XGBoost, adapted for
a sequence model. LSTM input is (samples, timesteps, features),
so a few things work differently from the tree-based version:

  - shap.GradientExplainer / DeepExplainer on recurrent (LSTM) Keras
    models is a long-standing, unresolved compatibility problem in the
    shap library itself (see shap/shap issues #1083, #1122, #1677,
    #2526, #3593) - it fails with shape-broadcast errors on many
    TF/Keras version combinations, through no fault of the model or
    data. So this script uses shap.KernelExplainer instead, which
    treats the model as a black box (just calls model.predict) and
    therefore sidesteps the gradient-tracing bug entirely. It holds
    everything except the LAST timestep fixed at a representative
    background sequence, and varies only the last timestep's features
    - which is exactly the slice the summary/bar plots are meant to
    show anyway ("how did each feature push THIS prediction, given
    recent history").

  - Feature importance (reports/feature_importance/lstm/) is instead
    computed via permutation importance, aggregated across the FULL
    24-step window - this is model-agnostic too, and is the closest
    LSTM analogue to XGBoost's built-in feature_importances_.

Run from the project root:
    python src/shap_analysis_lstm.py
===========================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import load_model

from lstm_utils import load_data, scale_data, create_sequences

# ==========================================================
# Directories
# ==========================================================

FEATURE_DIR = "data/features"
MODEL_DIR = "models/lstm"

FEATURE_IMPORTANCE_DIR = "reports/feature_importance/lstm"
SHAP_DIR = "reports/shap/lstm"

os.makedirs(FEATURE_IMPORTANCE_DIR, exist_ok=True)
os.makedirs(SHAP_DIR, exist_ok=True)

regions = [
    "northern",
    "western",
    "eastern",
    "southern",
    "northeastern",
]

SEQUENCE_LENGTH = 24

# Sample sizes (keep small - GradientExplainer on sequences is not cheap)
N_BACKGROUND = 50
N_EXPLAIN = 200

print("=" * 70)
print("LSTM SHAP + FEATURE IMPORTANCE ANALYSIS")
print("=" * 70)

for region in regions:

    print(f"\nProcessing {region.upper()}")

    # ------------------------------------------------------
    # Reproduce identical data prep from train_lstm.py
    # ------------------------------------------------------

    feature_file = os.path.join(FEATURE_DIR, f"{region}_features.csv")
    df = load_data(feature_file)

    (
        X_scaled,
        y_scaled,
        feature_scaler,
        target_scaler,
        feature_names,
    ) = scale_data(df)

    X_seq, y_seq = create_sequences(
        X_scaled, y_scaled, sequence_length=SEQUENCE_LENGTH
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y_seq, test_size=0.2, shuffle=False
    )

    model_file = os.path.join(MODEL_DIR, f"{region}_lstm.keras")
    model = load_model(model_file)

    # ------------------------------------------------------
    # Sample background + explain sets
    # ------------------------------------------------------

    rng = np.random.default_rng(42)

    bg_idx = rng.choice(len(X_train), size=min(N_BACKGROUND, len(X_train)), replace=False)
    background = X_train[bg_idx]

    ex_idx = rng.choice(len(X_test), size=min(N_EXPLAIN, len(X_test)), replace=False)
    X_explain = X_test[ex_idx]
    y_explain = y_test[ex_idx]

    n_features = len(feature_names)

    # ------------------------------------------------------
    # Feature importance: permutation importance, aggregated
    # across the full sequence window. Model-agnostic, so it
    # always works regardless of shap/TF/Keras versions.
    # ------------------------------------------------------

    baseline_pred = model.predict(X_explain, verbose=0).flatten()
    baseline_mae = mean_absolute_error(y_explain.flatten(), baseline_pred)

    importances = np.zeros(n_features)

    for j in range(n_features):
        X_perm = X_explain.copy()
        shuffled = X_perm[:, :, j].copy()
        rng.shuffle(shuffled)
        X_perm[:, :, j] = shuffled

        perm_pred = model.predict(X_perm, verbose=0).flatten()
        perm_mae = mean_absolute_error(y_explain.flatten(), perm_pred)

        importances[j] = perm_mae - baseline_mae

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances,
    }).sort_values(by="Importance", ascending=False)

    importance_df.to_csv(
        os.path.join(FEATURE_IMPORTANCE_DIR, f"{region}_importance.csv"),
        index=False,
    )

    plt.figure(figsize=(8, 7))
    plt.barh(
        importance_df["Feature"][:15][::-1],
        importance_df["Importance"][:15][::-1],
    )
    plt.title(f"{region.title()} Feature Importance (LSTM, permutation)")
    plt.xlabel("Increase in MAE when feature is shuffled")
    plt.tight_layout()
    plt.savefig(
        os.path.join(FEATURE_IMPORTANCE_DIR, f"{region}_importance.png"),
        dpi=300,
    )
    plt.close()

    print("  Permutation importance done -> reports/feature_importance/lstm/")

    # ------------------------------------------------------
    # SHAP plots: shap.KernelExplainer, black-box (calls
    # model.predict only), so it sidesteps the recurrent-model
    # gradient-tracing bug entirely. Only the LAST timestep is
    # varied; everything before it is held at a representative
    # background sequence, so KernelExplainer only has to deal
    # with n_features inputs, not seq_len * n_features.
    # ------------------------------------------------------

    try:
        import shap

        # A single representative "history" sequence: mean of the
        # background sequences, used to fill in timesteps 0..-2
        # while the last timestep is varied per explained instance.
        background_seq = background.mean(axis=0)  # (seq_len, n_features)

        def predict_last_timestep(last_step_batch):
            n = last_step_batch.shape[0]
            seqs = np.repeat(background_seq[np.newaxis, :, :], n, axis=0)
            seqs[:, -1, :] = last_step_batch
            return model.predict(seqs, verbose=0).flatten()

        background_last = background[:, -1, :]
        kmeans_bg = shap.kmeans(background_last, min(15, len(background_last)))

        # Keep this small - KernelExplainer calls predict_last_timestep
        # many times per explained instance.
        N_EXPLAIN_KERNEL = 60
        kex_idx = rng.choice(len(X_explain), size=min(N_EXPLAIN_KERNEL, len(X_explain)), replace=False)
        X_last_explain = X_explain[kex_idx][:, -1, :]

        kernel_explainer = shap.KernelExplainer(predict_last_timestep, kmeans_bg)
        shap_last = kernel_explainer.shap_values(X_last_explain, nsamples=100, silent=True)

        if isinstance(shap_last, list):
            shap_last = shap_last[0]
        shap_last = np.array(shap_last)

        plt.figure()
        shap.summary_plot(
            shap_last, X_last_explain, feature_names=feature_names, show=False
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(SHAP_DIR, f"{region}_summary.png"),
            dpi=300, bbox_inches="tight",
        )
        plt.close()

        plt.figure()
        shap.summary_plot(
            shap_last, X_last_explain, feature_names=feature_names,
            plot_type="bar", show=False,
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(SHAP_DIR, f"{region}_bar.png"),
            dpi=300, bbox_inches="tight",
        )
        plt.close()

        print("  SHAP KernelExplainer succeeded -> reports/shap/lstm/")

    except Exception as e:
        print(f"  SHAP KernelExplainer failed ({type(e).__name__}: {e})")
        print("  reports/shap/lstm/ will NOT be updated for this region.")

print("\n" + "=" * 70)
print("LSTM SHAP + Feature Importance Analysis Completed")
print("Reports generated inside reports/feature_importance/lstm and reports/shap/lstm")
print("=" * 70)
