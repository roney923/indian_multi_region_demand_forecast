# Eastern India Electricity Demand Forecasting

Short-term (1h/6h/24h) electricity demand forecasting for India's Eastern Region
(West Bengal, Odisha, Bihar, Jharkhand) using weather data and machine learning.

Research project for [your course name], [your institution].

## Motivation

India faces recurring electricity shortages and blackouts, particularly during
seasonal demand peaks. This project explores whether weather-informed ML models
can meaningfully improve short-term demand forecasting over naive persistence
baselines, using real grid operations data from India's National Load Despatch
Centre (NLDC).

## Acknowledgements & Prior Work

This project's methodology (lag/rolling feature engineering, LSTM/XGBoost
comparison against naive persistence) was inspired by the open-source
[AI_Energy_Manager](https://github.com/mehdighelich/AI_Energy_Manager) project
by Mehdi Ghelich (MIT License), which applied a similar pipeline to Spanish
(Madrid) load and weather data. This project extends that approach to Indian
grid data, with the following differences:

- Uses real Indian grid operations data (NLDC/Grid-India via the
  [India Power Grid — NLDC Daily PSP Reports](https://www.kaggle.com/datasets/halcyonvector/india-power-grid-nldc-daily-psp-reports)
  Kaggle dataset) instead of Spanish data
- Targets Eastern Region demand specifically (rather than national aggregate),
  matched against composite weather from four Eastern cities (Kolkata,
  Bhubaneswar, Patna, Ranchi) rather than a single city
- Compares XGBoost against naive persistence across three forecast horizons,
  with an explicit discussion of where each approach wins/loses

Weather data sourced from the free [Open-Meteo Historical Weather API](https://open-meteo.com/).

## Data Sources

| Source | What | License |
|---|---|---|
| NLDC / Grid-India (via Kaggle) | Hourly regional electricity demand, generation mix | CC BY-SA 4.0 |
| Open-Meteo | Hourly temperature, humidity, wind, cloud cover | CC BY 4.0 |

## Pipeline

1. `src/merge_data.py` — merges regional demand data with multi-city weather data
2. `src/features.py` — builds lag/rolling/time-based features
3. `src/train.py` — trains XGBoost models for 1h/6h/24h horizons, compares
   against naive persistence baseline

## Results

See `results/eastern_results.json` for full metrics. Summary:

| Horizon | Best Model | MAE (MW) | R² |
|---|---|---|---|
| 1h  | XGBoost | 445 | 0.925 |
| 6h  | XGBoost | 693 | 0.812 |
| 24h | Naive   | 645 | 0.812 |

XGBoost clearly outperforms naive persistence at 1h and 6h horizons. At 24h,
naive persistence (same hour, previous day) is competitive or slightly better,
suggesting the current feature set does not yet capture enough additional
signal beyond daily seasonality at that horizon — a direction for future work
(see below).

## Future Work

- Improve 24h-ahead performance (additional features: holidays, day-ahead
  weather forecasts rather than same-day weather)
- Add LSTM comparison alongside XGBoost
- SHAP-based feature importance analysis
- Extend to other Grid-India regions (Northern, Western, Southern) for a
  full regional comparison

## Setup

```bash
pip install -r requirements.txt
python src/merge_data.py
python src/features.py
python src/train.py
```
