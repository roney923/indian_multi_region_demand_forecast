import os
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from xgboost import XGBRegressor

# =====================================
# Directories
# =====================================

FEATURE_DIR = "data/features"
MODEL_DIR = "models"
RESULT_DIR = "results"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

REGIONS = [
    "northern",
    "western",
    "eastern",
    "southern",
    "northeastern"
]

summary = []

# =====================================
# Train every region
# =====================================

for region in REGIONS:

    print("\n" + "=" * 70)
    print(f"Training {region.upper()} Region")
    print("=" * 70)

    file = f"{FEATURE_DIR}/{region}_features.csv"

    df = pd.read_csv(file)

    # -------------------------------
    # Target
    # -------------------------------

    y = df["load"]

    X = df.drop(columns=["load", "datetime"])

    # -------------------------------
    # Time Split
    # -------------------------------

    split = int(len(df) * 0.8)

    X_train = X.iloc[:split]
    X_test = X.iloc[split:]

    y_train = y.iloc[:split]
    y_test = y.iloc[split:]

    # =====================================
    # Naive Forecast
    # =====================================

    naive_pred = X_test["lag_1"]

    # =====================================
    # XGBoost
    # =====================================

    model = XGBRegressor(

        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42

    )

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    # =====================================
    # Metrics
    # =====================================

    mae = mean_absolute_error(y_test, pred)

    rmse = np.sqrt(mean_squared_error(y_test, pred))

    mape = np.mean(np.abs((y_test - pred) / y_test)) * 100

    r2 = r2_score(y_test, pred)

    naive_mae = mean_absolute_error(y_test, naive_pred)

    # =====================================
    # Save Model
    # =====================================

    joblib.dump(
        model,
        f"{MODEL_DIR}/{region}_xgboost.joblib"
    )

    # =====================================
    # Save Predictions
    # =====================================

    out = pd.DataFrame({

        "Actual": y_test,

        "Naive": naive_pred,

        "XGBoost": pred

    })

    out.to_csv(

        f"{RESULT_DIR}/{region}_predictions.csv",

        index=False

    )

    summary.append({

        "Region": region,

        "Naive MAE": round(naive_mae,2),

        "XGBoost MAE": round(mae,2),

        "RMSE": round(rmse,2),

        "MAPE": round(mape,2),

        "R2": round(r2,4)

    })

    print(f"Naive MAE : {naive_mae:.2f}")

    print(f"XGBoost MAE : {mae:.2f}")

    print(f"RMSE : {rmse:.2f}")

    print(f"MAPE : {mape:.2f}%")

    print(f"R² : {r2:.4f}")

# =====================================
# Summary
# =====================================

summary_df = pd.DataFrame(summary)

summary_df.to_csv(

    f"{RESULT_DIR}/model_comparison.csv",

    index=False

)

print("\n")
print("=" * 70)
print(summary_df)
print("=" * 70)

print("\nTraining Complete.")