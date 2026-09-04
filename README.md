# Rental Bike Demand Prediction 🚲

Predicting hourly rental bike demand in Seoul using weather, date, and time
features — with a full EDA-to-deployment workflow: cleaning, feature
engineering, model comparison, hyperparameter tuning, and a reusable
inference module for serving predictions on new inputs.

## Overview

Bike-sharing systems need to know how many bikes to have available at any
given hour to balance demand without over- or under-supplying stations. This
project builds a regression model that predicts the **count of rented bikes
per hour** from weather conditions, the season, holiday status, and
whether the day is a functioning day.

## Dataset

- **Source:** [Seoul Bike Sharing Demand dataset (UCI Machine Learning Repository)](https://archive.ics.uci.edu/dataset/560/seoul+bike+sharing+demand)
- **Size:** 8,760 hourly records (Dec 2017 – Nov 2018)
- **Target:** `Rented Bike Count`
- **Features:** Date, Hour, Temperature, Humidity, Wind speed, Visibility,
  Dew point temperature, Solar Radiation, Rainfall, Snowfall, Seasons,
  Holiday, Functioning Day

## Approach

1. **Data cleaning** — checked for nulls and inconsistent types.
2. **EDA** — distribution plots, seasonal/hourly demand trends, and
   correlation analysis.
3. **Skew correction** on numeric features.
4. **Multicollinearity check** — `Dew point temperature` was highly
   correlated with `Temperature` and was dropped (via VIF analysis).
5. **Feature encoding** — one-hot encoding for `Seasons` and weekday, binary
   encoding for `Holiday` / `Functioning Day`.
6. **Scaling** — `StandardScaler` applied to features before training.
7. **Model comparison** — Linear Regression, Decision Tree, Random Forest,
   and XGBoost were trained and evaluated on a held-out test set.
8. **Hyperparameter tuning** — `RandomizedSearchCV` used to tune Random
   Forest and XGBoost.
9. **Final model** — XGBoost Regressor, saved with its `StandardScaler` for
   inference.

## Results

| Model                  | MSE       | RMSE   | MAE    | R² |
|-------------------------|-----------|--------|--------|------|
| Linear Regression        | 186,569.98| 431.94 | 330.21 | 0.543 |
| Decision Tree Regressor  | 58,397.41 | 241.66 | 131.83 | 0.857 |
| Random Forest Regressor  | 30,104.30 | 173.51 | 98.41  | 0.926 |
| **XGBoost Regressor**    | **25,361.39** | **159.25** | **97.86** | **0.938** |

The final saved XGBoost model (`xgboost_regressor_r2_0_928_v1.pkl`) achieves
an **R² of ~0.93** on the test set.

## Project Structure

```
rental-bike-demand-prediction/
├── data/
│   └── RawData.csv
├── models/
│   ├── xgboost_regressor_r2_0_928_v1.pkl   # Trained XGBoost model
│   └── sc.pkl                              # Fitted StandardScaler
├── Rental_Bike_Demand.ipynb   # Full EDA, training & tuning workflow
├── Inference.ipynb
├── src/
│   └── inference.py            
├── requirements.txt
└── README.md
```

## Setup

```bash
git clone https://github.com/<your-username>/rental-bike-demand-prediction.git
cd rental-bike-demand-prediction
python -m venv venv
source venv/bin/activate    # venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Usage

### Explore the analysis

Open the notebook to walk through EDA, feature engineering, and model
training:

```bash
jupyter lab notebooks/Rental_Bike_Demand.ipynb
```

### Run a prediction

```bash
python src/inference.py
```

You'll be prompted for date, hour, and weather details, and the script will
print the predicted rental bike count. You can also import `Inference`
directly in your own code:

```python
from src.inference import Inference

model = Inference(
    model_path="models/xgboost_regressor_r2_0_928_v1.pkl",
    sc_path="models/sc.pkl",
)

features = model.build_features(
    date="15/06/2018", hour=8, temperature=22.5, humidity=55,
    wind_speed=1.5, visibility=2000, solar_radiation=0.5,
    rainfall=0, snowfall=0, season="Summer",
    holiday="No Holiday", functioning_day="Yes",
)

print(model.predict(features))
```

## Tech Stack

Python · pandas · NumPy · scikit-learn · XGBoost · statsmodels · Matplotlib
· Seaborn · Jupyter

## License

This project is released under the [MIT License](LICENSE).
