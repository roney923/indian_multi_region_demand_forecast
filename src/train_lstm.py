"""
===========================================================
Train LSTM Models
Indian Multi-Region Electricity Demand Forecasting

PART 1
-------------------------------------
✓ Directory Creation
✓ Region Loop
✓ Load Dataset
✓ Scaling
✓ Sequence Generation
✓ Train/Test Split
===========================================================
"""

import os

from sklearn.model_selection import train_test_split

from lstm_utils import (
    load_data,
    scale_data,
    create_sequences
)

# ==========================================================
# Directory Structure
# ==========================================================

FEATURE_DIR = "data/features"

MODEL_DIR = "models/lstm"

RESULT_DIR = "results/lstm"

REPORT_DIR = "reports/training_history"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# ==========================================================
# Regions
# ==========================================================

regions = [
    "northern",
    "western",
    "eastern",
    "southern",
    "northeastern"
]

# ==========================================================
# Sequence Length
# ==========================================================

SEQUENCE_LENGTH = 24

print("=" * 70)
print("LSTM TRAINING PIPELINE")
print("=" * 70)

# ==========================================================
# Process Every Region
# ==========================================================

for region in regions:

    print("\n" + "=" * 70)
    print(f"Processing {region.upper()} Region")
    print("=" * 70)

    # ------------------------------------------------------
    # Load Dataset
    # ------------------------------------------------------

    feature_file = os.path.join(
        FEATURE_DIR,
        f"{region}_features.csv"
    )

    print("Loading dataset...")

    df = load_data(feature_file)

    print(f"Dataset Shape : {df.shape}")

    # ------------------------------------------------------
    # Scale Dataset
    # ------------------------------------------------------

    print("Scaling features...")

    (
        X_scaled,
        y_scaled,
        feature_scaler,
        target_scaler,
        feature_names
    ) = scale_data(df)

    print(f"Features : {len(feature_names)}")

    # ------------------------------------------------------
    # Create Sequences
    # ------------------------------------------------------

    print("Creating sequences...")

    X_seq, y_seq = create_sequences(
        X_scaled,
        y_scaled,
        sequence_length=SEQUENCE_LENGTH
    )

    print(f"Sequence Shape : {X_seq.shape}")
    print(f"Target Shape   : {y_seq.shape}")

    # ------------------------------------------------------
    # Train/Test Split
    # ------------------------------------------------------

    print("Splitting dataset...")

    X_train, X_test, y_train, y_test = train_test_split(
        X_seq,
        y_seq,
        test_size=0.2,
        shuffle=False
    )

    print("\nDataset Summary")

    print(f"Training Samples : {len(X_train):,}")
    print(f"Testing Samples  : {len(X_test):,}")

    print(f"Input Shape      : {X_train.shape}")
    print(f"Target Shape     : {y_train.shape}")

    # ------------------------------------------------------
    # Placeholder
    # ------------------------------------------------------

    print("\n✓ Data preparation completed.")

print("\n" + "=" * 70)
print("PART 1 COMPLETED SUCCESSFULLY")
print("=" * 70)