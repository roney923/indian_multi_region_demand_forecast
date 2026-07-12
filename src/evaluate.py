import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
import numpy as np

# =====================================================
# Directories
# =====================================================

RESULT_DIR = "results"
MODEL_DIR = "models"
FEATURE_DIR = "data/features"

REPORT_DIR = "reports"

PLOT_DIR = os.path.join(REPORT_DIR, "plots")
FEATURE_IMPORTANCE_DIR = os.path.join(REPORT_DIR, "feature_importance")
COMPARISON_DIR = os.path.join(REPORT_DIR, "comparison")
METRIC_DIR = os.path.join(REPORT_DIR, "metrics")
RESIDUAL_DIR = os.path.join(REPORT_DIR, "residuals")
ERROR_DIR = os.path.join(REPORT_DIR, "error_distribution")

os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(FEATURE_IMPORTANCE_DIR, exist_ok=True)
os.makedirs(COMPARISON_DIR, exist_ok=True)
os.makedirs(METRIC_DIR, exist_ok=True)
os.makedirs(RESIDUAL_DIR, exist_ok=True)
os.makedirs(ERROR_DIR, exist_ok=True)

# =====================================================
# Regions
# =====================================================

regions = [
    "northern",
    "western",
    "eastern",
    "southern",
    "northeastern"
]

summary = []

print("="*70)
print("MODEL EVALUATION")
print("="*70)

# =====================================================
# Evaluate Every Region
# =====================================================

for region in regions:

    print(f"\nEvaluating {region.upper()}")

    prediction_file = os.path.join(
        RESULT_DIR,
        f"{region}_predictions.csv"
    )

    feature_file = os.path.join(
        FEATURE_DIR,
        f"{region}_features.csv"
    )

    model_file = os.path.join(
        MODEL_DIR,
        f"{region}_xgboost.joblib"
    )

    pred = pd.read_csv(prediction_file)

    feature_df = pd.read_csv(feature_file)

    model = joblib.load(model_file)

    # ---------------------------------------------------
    # Metrics
    # ---------------------------------------------------

    mae = mean_absolute_error(
        pred["Actual"],
        pred["XGBoost"]
    )

    rmse = np.sqrt(
        mean_squared_error(
            pred["Actual"],
            pred["XGBoost"]
        )
    )

    mape = np.mean(
        np.abs(
            (pred["Actual"]-pred["XGBoost"])/pred["Actual"]
        )
    )*100

    r2 = r2_score(
        pred["Actual"],
        pred["XGBoost"]
    )

    summary.append({
        "Region": region,
        "MAE": round(mae,2),
        "RMSE": round(rmse,2),
        "MAPE": round(mape,2),
        "R2": round(r2,4)
    })

    # =====================================================
    # Prediction Plot
    # =====================================================

    plt.figure(figsize=(15,5))

    plt.plot(
        pred["Actual"][:500],
        label="Actual",
        linewidth=2
    )

    plt.plot(
        pred["XGBoost"][:500],
        label="Prediction",
        linewidth=2
    )

    plt.title(
        f"{region.title()} Region Demand Forecast"
    )

    plt.xlabel("Hour")

    plt.ylabel("Demand (MW)")

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOT_DIR,
            f"{region}_prediction.png"
        ),
        dpi=300
    )

    plt.close()

    # =====================================================
    # Feature Importance
    # =====================================================

    importance = pd.DataFrame({
        "Feature": model.feature_names_in_,
        "Importance": model.feature_importances_
    })

    importance = importance.sort_values(
        by="Importance",
        ascending=False
    )

    importance.to_csv(
        os.path.join(
            FEATURE_IMPORTANCE_DIR,
            f"{region}_importance.csv"
        ),
        index=False
    )

    plt.figure(figsize=(8,7))

    plt.barh(
        importance["Feature"][:15],
        importance["Importance"][:15]
    )

    plt.gca().invert_yaxis()

    plt.title(
        f"{region.title()} Feature Importance"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            FEATURE_IMPORTANCE_DIR,
            f"{region}_importance.png"
        ),
        dpi=300
    )

    plt.close()

    # =====================================================
    # Residual Analysis
    # =====================================================
    pred["Residual"] = pred["Actual"] - pred["XGBoost"]
    
    plt.figure(figsize=(12,5))
    plt.plot(
        pred["Residual"][:500],
        linewidth=1.5
    )
    plt.axhline(
        y=0,
        color="red",
        linestyle="--"
    )
    plt.title(
        f"{region.title()} Residual Plot"
    )
    plt.xlabel("Hour")
    plt.ylabel("Prediction Error (MW)")
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            RESIDUAL_DIR,
            f"{region}_residual.png"
        ),
        dpi=300
    )
    plt.close()
    
    # =====================================================
    # Error Distribution
    # =====================================================
    plt.figure(figsize=(8,5))
    plt.hist(
        pred["Residual"],
        bins=40
    )
    plt.title(
        f"{region.title()} Error Distribution"
    )
    plt.xlabel("Prediction Error (MW)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            ERROR_DIR,
            f"{region}_error_hist.png"
        ),
        dpi=300
    )
    plt.close()
    
    # =====================================================
    # Residual Statistics
    # =====================================================
    stats = pd.DataFrame({
        "Statistic":[
            "Mean Error",
            "Std Error",
            "Maximum Error",
            "Minimum Error"
        ],
        "Value":[
            pred["Residual"].mean(),
            pred["Residual"].std(),
            pred["Residual"].max(),
            pred["Residual"].min()
        ]
    })
    stats.to_csv(
        os.path.join(
            METRIC_DIR,
            f"{region}_residual_statistics.csv"
        ),
        index=False
    )

# =====================================================
# Summary
# =====================================================

summary = pd.DataFrame(summary)

summary.to_csv(
    os.path.join(
        METRIC_DIR,
        "model_summary.csv"
    ),
    index=False
)

# =====================================================
# MAE Comparison
# =====================================================

plt.figure(figsize=(9,5))

plt.bar(
    summary["Region"],
    summary["MAE"]
)

plt.ylabel("MAE")

plt.title("Region-wise MAE")

plt.tight_layout()

plt.savefig(
    os.path.join(
        COMPARISON_DIR,
        "mae_comparison.png"
    ),
    dpi=300
)

plt.close()

# =====================================================
# R² Comparison
# =====================================================

plt.figure(figsize=(9,5))

plt.bar(
    summary["Region"],
    summary["R2"]
)

plt.ylim(0,1)

plt.ylabel("R²")

plt.title("Region-wise R² Score")

plt.tight_layout()

plt.savefig(
    os.path.join(
        COMPARISON_DIR,
        "r2_comparison.png"
    ),
    dpi=300
)

plt.close()

print("\n")
print("="*70)
print(summary)
print("="*70)
print("\nEvaluation Completed Successfully!")
print("Reports generated inside /reports")