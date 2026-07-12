"""
===========================================================
LSTM Utility Functions
Indian Multi-Region Electricity Demand Forecasting
===========================================================
"""

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam


# ==========================================================
# Load Dataset
# ==========================================================

def load_data(filepath):
    """
    Load feature-engineered dataset.

    Parameters
    ----------
    filepath : str

    Returns
    -------
    df : pandas.DataFrame
    """

    df = pd.read_csv(filepath)

    # Remove datetime if present
    if "datetime" in df.columns:
        df = df.drop(columns=["datetime"])

    return df


# ==========================================================
# Scale Features
# ==========================================================

def scale_data(df, target_column="load"):
    """
    Scale features and target separately.

    Returns
    -------
    X_scaled
    y_scaled
    feature_scaler
    target_scaler
    """

    X = df.drop(columns=[target_column])

    y = df[target_column].values.reshape(-1, 1)

    feature_scaler = MinMaxScaler()

    target_scaler = MinMaxScaler()

    X_scaled = feature_scaler.fit_transform(X)

    y_scaled = target_scaler.fit_transform(y)

    return (
        X_scaled,
        y_scaled,
        feature_scaler,
        target_scaler,
        X.columns.tolist()
    )


# ==========================================================
# Create Sequences
# ==========================================================

def create_sequences(X, y, sequence_length=24):
    """
    Convert hourly data into sequences.

    Example

    Hour1
    Hour2
    ...
    Hour24

    --> Predict Hour25
    """

    X_seq = []
    y_seq = []

    for i in range(sequence_length, len(X)):

        X_seq.append(
            X[i-sequence_length:i]
        )

        y_seq.append(
            y[i]
        )

    X_seq = np.array(X_seq)

    y_seq = np.array(y_seq)

    return X_seq, y_seq


# ==========================================================
# Build LSTM Model
# ==========================================================

def build_lstm_model(input_shape):
    """
    Build stacked LSTM model.
    """

    model = Sequential()

    model.add(

        LSTM(

            64,

            return_sequences=True,

            input_shape=input_shape

        )

    )

    model.add(

        Dropout(0.2)

    )

    model.add(

        LSTM(

            32

        )

    )

    model.add(

        Dropout(0.2)

    )

    model.add(

        Dense(

            16,

            activation="relu"

        )

    )

    model.add(

        Dense(

            1

        )

    )

    model.compile(

        optimizer=Adam(

            learning_rate=0.001

        ),

        loss="mse",

        metrics=["mae"]

    )

    return model
# ==========================================================
# Evaluate Model
# ==========================================================

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import matplotlib.pyplot as plt


def evaluate_model(y_true, y_pred):
    """
    Calculate regression metrics.

    Returns
    -------
    metrics : dict
    """

    mae = mean_absolute_error(y_true, y_pred)

    rmse = np.sqrt(
        mean_squared_error(y_true, y_pred)
    )

    mape = np.mean(
        np.abs((y_true - y_pred) / y_true)
    ) * 100

    r2 = r2_score(
        y_true,
        y_pred
    )

    metrics = {

        "MAE": round(float(mae), 2),

        "RMSE": round(float(rmse), 2),

        "MAPE": round(float(mape), 2),

        "R2": round(float(r2), 4)

    }

    return metrics


# ==========================================================
# Plot Training History
# ==========================================================

def plot_training_history(
        history,
        region,
        save_dir="reports/training_history"
):
    """
    Plot training and validation loss.
    """

    import os

    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(8,5))

    plt.plot(
        history.history["loss"],
        label="Training Loss",
        linewidth=2
    )

    if "val_loss" in history.history:

        plt.plot(
            history.history["val_loss"],
            label="Validation Loss",
            linewidth=2
        )

    plt.title(
        f"{region.title()} LSTM Training History"
    )

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.legend()

    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            save_dir,
            f"{region}_training_loss.png"
        ),
        dpi=300
    )

    plt.close()


# ==========================================================
# Save Prediction CSV
# ==========================================================

def save_predictions(
        actual,
        predicted,
        region,
        save_dir="results/lstm"
):
    """
    Save actual vs predicted values.
    """

    import os

    os.makedirs(save_dir, exist_ok=True)

    prediction_df = pd.DataFrame({

        "Actual": actual.flatten(),

        "Prediction": predicted.flatten()

    })

    prediction_df.to_csv(

        os.path.join(

            save_dir,

            f"{region}_predictions.csv"

        ),

        index=False

    )


# ==========================================================
# Save Metrics
# ==========================================================

def save_metrics(
        metrics,
        region,
        save_dir="results/lstm"
):
    """
    Save metrics for one region.
    """

    import os

    os.makedirs(save_dir, exist_ok=True)

    metric_df = pd.DataFrame([metrics])

    metric_df.insert(
        0,
        "Region",
        region
    )

    metric_df.to_csv(

        os.path.join(

            save_dir,

            f"{region}_metrics.csv"

        ),

        index=False

    )