"""
===========================================================
Evaluate Trained LSTM Models & Generate Reports
Indian Multi-Region Electricity Demand Forecasting

This script assumes the LSTM models in models/lstm/*.keras
are ALREADY TRAINED (they are, per your models folder).
It re-creates the exact same data prep used in train_lstm.py
(same code path -> same deterministic scaling/split, since
shuffle=False and no randomness is involved), loads each
saved model, generates predictions, and writes out results
+ reports in the same style as your XGBoost evaluate.py.

Run this from the project root:
    cd indian_multi_region_demand_forecast
    python src/evaluate_lstm.py
===========================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model

from lstm_utils import (
    load_data,
    scale_data,
    create_sequences,
    evaluate_model,
    save_predictions,
    save_metrics,
)

# ==========================================================
# Directories
# ==========================================================

FEATURE_DIR = "data/features"
MODEL_DIR = "models/lstm"
RESULT_DIR = "results/lstm"

REPORT_DIR = "reports"
PLOT_DIR = os.path.join(REPORT_DIR, "plots", "lstm")
COMPARISON_DIR = os.path.join(REPORT_DIR, "comparison", "lstm")
METRIC_DIR = os.path.join(REPORT_DIR, "metrics", "lstm")
RESIDUAL_DIR = os.path.join(REPORT_DIR, "residuals", "lstm")
ERROR_DIR = os.path.join(REPORT_DIR, "error_distribution", "lstm")

for d in [RESULT_DIR, PLOT_DIR, COMPARISON_DIR, METRIC_DIR, RESIDUAL_DIR, ERROR_DIR]:
    os.makedirs(d, exist_ok=True)

# ==========================================================
# Regions
# ==========================================================

regions = [
    "northern",
    "western",
    "eastern",
    "southern",
    "northeastern",
]

SEQUENCE_LENGTH = 24

summary = []

print("=" * 70)
print("LSTM MODEL EVALUATION")
print("=" * 70)

for region in regions:

    print(f"\nEvaluating {region.upper()}")

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

    # ------------------------------------------------------
    # Load trained model & predict
    # ------------------------------------------------------

    model_file = os.path.join(MODEL_DIR, f"{region}_lstm.keras")
    model = load_model(model_file)

    y_pred_scaled = model.predict(X_test, verbose=0)

    # Inverse-transform back to real demand units (MW)
    y_test_actual = target_scaler.inverse_transform(y_test)
    y_pred_actual = target_scaler.inverse_transform(y_pred_scaled)

    # ------------------------------------------------------
    # Save predictions
    # ------------------------------------------------------

    save_predictions(y_test_actual, y_pred_actual, region, save_dir=RESULT_DIR)

    # ------------------------------------------------------
    # Metrics
    # ------------------------------------------------------

    metrics = evaluate_model(y_test_actual.flatten(), y_pred_actual.flatten())
    # Save once for the dashboard (results/lstm/{region}_metrics.csv, matches
    # what dashboard/app.py looks for) and once alongside the other report
    # artifacts (reports/metrics/lstm/{region}_metrics.csv).
    save_metrics(metrics, region, save_dir=RESULT_DIR)
    save_metrics(metrics, region, save_dir=METRIC_DIR)

    summary.append({"Region": region, **metrics})

    # ------------------------------------------------------
    # Prediction plot
    # ------------------------------------------------------

    plt.figure(figsize=(15, 5))
    plt.plot(y_test_actual[:500], label="Actual", linewidth=2)
    plt.plot(y_pred_actual[:500], label="Prediction", linewidth=2)
    plt.title(f"{region.title()} Region Demand Forecast (LSTM)")
    plt.xlabel("Hour")
    plt.ylabel("Demand (MW)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{region}_prediction.png"), dpi=300)
    plt.close()

    # ------------------------------------------------------
    # Residual analysis
    # ------------------------------------------------------

    residual = y_test_actual.flatten() - y_pred_actual.flatten()

    plt.figure(figsize=(12, 5))
    plt.plot(residual[:500], linewidth=1.5)
    plt.axhline(y=0, color="red", linestyle="--")
    plt.title(f"{region.title()} Residual Plot (LSTM)")
    plt.xlabel("Hour")
    plt.ylabel("Prediction Error (MW)")
    plt.tight_layout()
    plt.savefig(os.path.join(RESIDUAL_DIR, f"{region}_residual.png"), dpi=300)
    plt.close()

    # ------------------------------------------------------
    # Error distribution
    # ------------------------------------------------------

    plt.figure(figsize=(8, 5))
    plt.hist(residual, bins=40)
    plt.title(f"{region.title()} Error Distribution (LSTM)")
    plt.xlabel("Prediction Error (MW)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(ERROR_DIR, f"{region}_error_hist.png"), dpi=300)
    plt.close()

    print(f"  MAE={metrics['MAE']}  RMSE={metrics['RMSE']}  "
          f"MAPE={metrics['MAPE']}%  R2={metrics['R2']}")

# ==========================================================
# Summary
# ==========================================================

summary_df = pd.DataFrame(summary)
summary_df.to_csv(os.path.join(METRIC_DIR, "model_summary.csv"), index=False)

# MAE comparison
plt.figure(figsize=(9, 5))
plt.bar(summary_df["Region"], summary_df["MAE"])
plt.ylabel("MAE")
plt.title("Region-wise MAE (LSTM)")
plt.tight_layout()
plt.savefig(os.path.join(COMPARISON_DIR, "mae_comparison.png"), dpi=300)
plt.close()

# R2 comparison
plt.figure(figsize=(9, 5))
plt.bar(summary_df["Region"], summary_df["R2"])
plt.ylim(0, 1)
plt.ylabel("R2")
plt.title("Region-wise R2 Score (LSTM)")
plt.tight_layout()
plt.savefig(os.path.join(COMPARISON_DIR, "r2_comparison.png"), dpi=300)
plt.close()

print("\n" + "=" * 70)
print(summary_df)
print("=" * 70)
print("\nLSTM Evaluation Completed Successfully!")
print("Reports generated inside /reports/*/lstm and /results/lstm")