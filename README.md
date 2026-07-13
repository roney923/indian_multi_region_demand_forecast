# GridCast AI — Multi-Region Electricity Demand Forecasting for the Indian National Grid

Machine learning pipeline forecasting hourly electricity demand across India's five power grid regions (Northern, Western, Eastern, Southern, North-Eastern), using historical demand + weather data, with model explainability and an interactive dashboard.

**Live dashboard:** [electricityforcast96922.streamlit.app](https://electricityforcast96922.streamlit.app/)

---

## Motivation

India faces recurring electricity shortages and demand-supply mismatches, particularly during seasonal peaks. This project explores whether weather-informed machine learning models can meaningfully improve short-term regional demand forecasting over a naive persistence baseline, using real grid operations data from India's National Load Despatch Centre (NLDC), across all five regional grids rather than a single region.

## Pipeline Overview

```
Weather Data (Open-Meteo)  +  Demand Data (NLDC / Grid-India)
                    │
                    ▼
              merge_data.py
                    │
                    ▼
              features.py           (lag, rolling, calendar features)
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
    train.py                train_lstm.py
    (XGBoost)                (LSTM)
        │                       │
        ▼                       ▼
    evaluate.py             evaluate_lstm.py
    shap_analysis.py        shap_analysis_lstm.py
        │                       │
        └───────────┬───────────┘
                     ▼
            dashboard/app.py
         (Streamlit — deployed)
```

A full architecture diagram is in [`docs/architecture.png`](docs/architecture.png).

## Repository Structure

```
indian_multi_region_demand_forecast/
├── dashboard/
│   └── app.py                  # Streamlit dashboard (region + model selector, metrics, SHAP, comparison)
├── data/
│   └── features/                # Feature-engineered datasets per region
├── docs/
│   ├── architecture.png
│   └── methodology.pdf
├── models/
│   ├── xgboost/                 # Trained XGBoost models per region
│   └── lstm/                    # Trained LSTM models + scalers per region
├── reports/
│   ├── comparison/               # XGBoost vs LSTM comparison charts
│   ├── feature_importance/       # Per-model feature importance
│   ├── metrics/                  # Per-model metrics + residual statistics
│   ├── plots/                    # Prediction plots
│   ├── residuals/                # Residual analysis plots
│   └── shap/                     # SHAP explainability plots
├── results/
│   ├── xgboost/                  # Predictions + comparison CSVs
│   └── lstm/                     # Predictions + metrics CSVs
├── src/
│   ├── download_weather.py       # Pulls historical weather from Open-Meteo
│   ├── merge_data.py             # Merges demand + weather per region
│   ├── features.py               # Lag/rolling/calendar feature engineering
│   ├── train.py                  # XGBoost training + naive baseline
│   ├── evaluate.py                # XGBoost evaluation, plots, comparison charts
│   ├── shap_analysis.py           # SHAP explainability (XGBoost)
│   ├── lstm_utils.py              # Shared LSTM helpers (scaling, sequencing, model)
│   ├── train_lstm.py              # LSTM training
│   ├── evaluate_lstm.py           # LSTM SHAP + permutation feature importance
│   └── shap_analysis_lstm.py
├── DATA_SOURCES.md
├── requirements.txt
└── README.md
```

## Data Sources

| Source | What | License |
|---|---|---|
| [NLDC / Grid-India PSP Reports](https://www.kaggle.com/datasets/halcyonvector/india-power-grid-nldc-daily-psp-reports) | Hourly regional electricity demand, generation mix | CC BY-SA 4.0 |
| [Indian Power Demand and Shortage Data](https://www.kaggle.com/datasets/preygle/indian-power-demand-and-shortage-data-2020-2025) | Supplementary demand/shortage data | — |
| [Open-Meteo](https://open-meteo.com/) | Hourly temperature, humidity, wind speed, cloud cover for 4 representative cities per region | CC BY 4.0 |

Full details in [`DATA_SOURCES.md`](DATA_SOURCES.md).

## Methodology

**Feature engineering** (`features.py`): hourly lag features (1h, 3h, 6h, 12h, 24h, 48h, 168h), rolling mean/std/max/min (6h, 24h windows), calendar features (hour, day, month, day-of-week, weekend flag), plus region-averaged and per-city weather variables.

**Models compared, per region:**
- **Naive persistence** — same hour, previous day
- **XGBoost** — gradient-boosted trees on the full tabular feature set
- **LSTM** — 2-layer stacked LSTM (64→32 units) on 16-hour sequence windows, trained with `EarlyStopping` + `ReduceLROnPlateau`

**Explainability:** SHAP summary/bar plots and feature importance for XGBoost (native); for LSTM, permutation importance (aggregated across the sequence window) plus SHAP `KernelExplainer` applied to the final timestep, since gradient-based SHAP explainers are not reliably compatible with recurrent Keras models.

## Results

**XGBoost vs. Naive Baseline** (test set, chronological 80/20 split):

| Region | Naive MAE | XGBoost MAE | RMSE | MAPE | R² |
|---|--:|--:|--:|--:|--:|
| Northern | 1923.96 | 931.05 | 1272.76 | 1.74% | 0.985 |
| Western | 1178.34 | 891.89 | 1246.06 | 1.47% | 0.957 |
| Eastern | 512.26 | 347.12 | 531.69 | 1.56% | 0.973 |
| Southern | 1301.32 | 767.01 | 1082.13 | 1.51% | 0.977 |
| North-Eastern | 97.89 | 45.21 | 62.69 | 2.04% | 0.980 |

**LSTM** (test set):

| Region | MAE | RMSE | MAPE | R² |
|---|--:|--:|--:|--:|
| Northern | 2645.88 | 3368.94 | 5.17% | 0.897 |
| Western | 2074.55 | 2669.78 | 3.45% | 0.803 |
| Eastern | 962.59 | 1187.95 | 4.57% | 0.868 |
| Southern | 2585.54 | 3463.34 | 5.28% | 0.766 |
| North-Eastern | 128.13 | 164.40 | 5.70% | 0.865 |

XGBoost outperforms LSTM across every region on this dataset — a common and expected result for tabular, lag-feature-rich data, where tree-based models often have an edge over sequence models unless the latter are given significantly more tuning, data, or architectural depth. Full comparison charts: [`reports/comparison/`](reports/comparison).

## Dashboard

An interactive Streamlit dashboard ([live demo](https://electricityforcast96922.streamlit.app/)) lets you:
- Select a region and a model (XGBoost or LSTM)
- View historical demand alongside predictions
- Inspect MAE / RMSE / MAPE / R² metrics
- Explore SHAP explainability and feature importance plots
- Compare XGBoost vs. LSTM performance side by side

Run locally:
```bash
streamlit run dashboard/app.py
```

## Installation

```bash
git clone https://github.com/roney923/indian_multi_region_demand_forecast.git
cd indian_multi_region_demand_forecast
pip install -r requirements.txt
pip install tensorflow   # required for LSTM training/evaluation, not in requirements.txt
```

## Usage

Run the pipeline stages in order (each stage reads the previous stage's output):

```bash
python src/download_weather.py     # fetch historical weather (one-time)
python src/merge_data.py           # merge demand + weather per region
python src/features.py             # build lag/rolling/calendar features

python src/train.py                # train XGBoost + naive baseline, all regions
python src/evaluate.py             # XGBoost plots, metrics, comparison charts
python src/shap_analysis.py        # SHAP explainability (XGBoost)

python src/train_lstm.py           # train LSTM, all regions
python src/evaluate_lstm.py        # LSTM SHAP + permutation feature importance

streamlit run dashboard/app.py     # launch the dashboard
```

## Future Work

- `compare_models.py` — head-to-head XGBoost vs. LSTM comparison including training time, inference time, and memory footprint (in progress)
- `predict.py` — forecasting on newly supplied weather/demand data, beyond historical backtesting (in progress)
- Holiday/festival demand features (Diwali, Holi, national holidays)
- Additional weather variables (rainfall, pressure, solar radiation)
- Rolling-origin time-series cross-validation in place of a single train/test split
- Hyperparameter tuning (Optuna/GridSearchCV) for both models

## Acknowledgements

This project's methodology (lag/rolling feature engineering, tree-based vs. sequence-model comparison against naive persistence) was inspired by the open-source [AI_Energy_Manager](https://github.com/mehdighelich/AI_Energy_Manager) project by Mehdi Ghelich (MIT License), which applied a similar pipeline to Spanish (Madrid) load and weather data. This project extends that approach to all five Indian Grid-India regions, using real NLDC grid operations data and region-specific multi-city weather composites.

## License

See [`LICENSE`](LICENSE).
