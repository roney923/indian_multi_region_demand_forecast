import os
import joblib
import shap
import pandas as pd
import matplotlib.pyplot as plt

# ===========================================
# Directories
# ===========================================

MODEL_DIR = "models"
FEATURE_DIR = "data/features"

SHAP_DIR = "reports/shap"

os.makedirs(SHAP_DIR, exist_ok=True)

regions = [
    "northern",
    "western",
    "eastern",
    "southern",
    "northeastern"
]

print("="*70)
print("SHAP ANALYSIS")
print("="*70)

for region in regions:

    print(f"\nProcessing {region.upper()}")

    feature_file = os.path.join(
        FEATURE_DIR,
        f"{region}_features.csv"
    )

    model_file = os.path.join(
        MODEL_DIR,
        f"{region}_xgboost.joblib"
    )

    df = pd.read_csv(feature_file)

    X = df.drop(columns=["datetime","load"])

    model = joblib.load(model_file)

    # ---------------------------------------
    # Use only first 500 rows
    # (much faster)
    # ---------------------------------------

    X_sample = X.iloc[:500]

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X_sample)

    # ===========================================
    # Summary Plot
    # ===========================================

    plt.figure()

    shap.summary_plot(

        shap_values,

        X_sample,

        show=False

    )

    plt.tight_layout()

    plt.savefig(

        os.path.join(

            SHAP_DIR,

            f"{region}_summary.png"

        ),

        dpi=300,

        bbox_inches="tight"

    )

    plt.close()

    # ===========================================
    # Bar Plot
    # ===========================================

    plt.figure()

    shap.summary_plot(

        shap_values,

        X_sample,

        plot_type="bar",

        show=False

    )

    plt.tight_layout()

    plt.savefig(

        os.path.join(

            SHAP_DIR,

            f"{region}_bar.png"

        ),

        dpi=300,

        bbox_inches="tight"

    )

    plt.close()

print("\nSHAP Analysis Completed Successfully!")