import os
import pandas as pd

# ==========================================
# Paths
# ==========================================

MERGED_DIR = "data/processed"
FEATURE_DIR = "data/features"

os.makedirs(FEATURE_DIR, exist_ok=True)

REGIONS = [
    "northern",
    "western",
    "eastern",
    "southern",
    "northeastern"
]

# ==========================================
# Feature Engineering Function
# ==========================================

def engineer_features(region):

    print("\n" + "=" * 70)
    print(f"Processing {region.upper()} Region")
    print("=" * 70)

    file_path = f"{MERGED_DIR}/{region}_merged.csv"

    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    df = pd.read_csv(file_path)

    df["datetime"] = pd.to_datetime(df["datetime"])

    df = df.sort_values("datetime")

    # ------------------------------------------------
    # Fill Missing Values
    # ------------------------------------------------

    df = df.ffill().bfill()

    # ------------------------------------------------
    # Time Features
    # ------------------------------------------------

    df["hour"] = df["datetime"].dt.hour
    df["day"] = df["datetime"].dt.day
    df["month"] = df["datetime"].dt.month
    df["day_of_week"] = df["datetime"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # ------------------------------------------------
    # Lag Features
    # ------------------------------------------------

    lag_hours = [1, 3, 6, 12, 24, 48, 168]

    for lag in lag_hours:
        df[f"lag_{lag}"] = df["load"].shift(lag)

    # ------------------------------------------------
    # Rolling Statistics
    # ------------------------------------------------

    df["rolling_mean_6"] = df["load"].rolling(6).mean()

    df["rolling_mean_24"] = df["load"].rolling(24).mean()

    df["rolling_std_24"] = df["load"].rolling(24).std()

    df["rolling_max_24"] = df["load"].rolling(24).max()

    df["rolling_min_24"] = df["load"].rolling(24).min()

    # ------------------------------------------------
    # Drop rows with lag NaNs
    # ------------------------------------------------

    rows_before = len(df)

    df = df.dropna()

    rows_after = len(df)

    removed = rows_before - rows_after

    # ------------------------------------------------
    # Save
    # ------------------------------------------------

    output_file = f"{FEATURE_DIR}/{region}_features.csv"

    df.to_csv(output_file, index=False)

    # ------------------------------------------------
    # Summary
    # ------------------------------------------------

    print(f"Rows before feature engineering : {rows_before:,}")
    print(f"Rows removed (lag creation)     : {removed:,}")
    print(f"Final rows                      : {rows_after:,}")
    print(f"Total columns                   : {len(df.columns)}")
    print(f"Missing values                  : {df.isna().sum().sum()}")

    print(f"\nSaved -> {output_file}")

# ==========================================
# Run
# ==========================================

for region in REGIONS:
    engineer_features(region)

print("\n" + "=" * 70)
print("FEATURE ENGINEERING COMPLETED")
print("=" * 70)