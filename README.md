# Implementation of Random Forest Algorithm for Weather Prediction
## AIM:
To write a program to predict daily temperature , PM2.5 pollution level and Energy based on environmental sensor data using Random Forest Algorithm.

## Problem Statement and Dataset
To develop a Random Forest model for predicting daily temperature, PM2.5 pollution level, and energy consumption using environmental sensor data.

Dataset:

The dataset contains environmental sensor readings such as temperature, humidity, air quality, PM2.5 levels, and energy usage. These features are used to train the Random Forest model for accurate prediction of environmental conditions and energy consumption.(Problem Statement and Dataset)


## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the environmental sensor dataset and preprocess missing values using Pandas.

2. Select environmental parameters as input features and prepare PM2.5, temperature, and energy consumption as target variables.

3. Train separate Decision Tree Regression models for pollution, temperature, and energy prediction using training data.

4. Predict the output values and evaluate model performance using RMSE and R² score metrics.

## Program:
```
/*
Program to implement the Random Forest Algorithm to predict daily temperature , PM2.5 pollution level and Energy based on environmental sensor data.
Developed by: Priyadharshini V
RegisterNumber: 212225230219



import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
data = pd.read_csv("weather-station-eee-block_2024_07_13.csv")

# Input features
X = data[['hum', 'pressure', 'wind_speed', 'illumination', 'co2']]

# Fill missing values in input features
X = X.fillna(X.mean())

# ---------------------------------------------------
# PM2.5 Pollution Prediction
# ---------------------------------------------------

y_pollution = data['pm2_5'].fillna(data['pm2_5'].mean())

X_train, X_test, y_train, y_test = train_test_split(
    X, y_pollution, test_size=0.2, random_state=42
)

pollution_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    max_depth=5
)

pollution_model.fit(X_train, y_train)

pollution_pred = pollution_model.predict(X_test)

rmse_pollution = np.sqrt(mean_squared_error(y_test, pollution_pred))
r2_pollution = r2_score(y_test, pollution_pred)
accuracy_pollution = r2_pollution * 100

print("🏭 Pollution Prediction (PM2.5)")
print("Accuracy (%):", accuracy_pollution)
print("RMSE:", rmse_pollution)
print("R2 Score:", r2_pollution)

# ---------------------------------------------------
# Temperature Prediction
# ---------------------------------------------------

y_temp = data['tem'].fillna(data['tem'].mean())

X_train, X_test, y_train, y_test = train_test_split(
    X, y_temp, test_size=0.2, random_state=42
)

temp_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    max_depth=5
)

temp_model.fit(X_train, y_train)

temp_pred = temp_model.predict(X_test)

rmse_temp = np.sqrt(mean_squared_error(y_test, temp_pred))
r2_temp = r2_score(y_test, temp_pred)
accuracy_temp = r2_temp * 100

print("\n🌡️ Temperature Prediction")
print("Accuracy (%):", accuracy_temp)
print("RMSE:", rmse_temp)
print("R2 Score:", r2_temp)

# ---------------------------------------------------
# Energy Prediction (TSR)
# ---------------------------------------------------

y_energy = data['tsr'].fillna(data['tsr'].mean())

X_train, X_test, y_train, y_test = train_test_split(
    X, y_energy, test_size=0.2, random_state=42
)

energy_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    max_depth=5
)

energy_model.fit(X_train, y_train)

energy_pred = energy_model.predict(X_test)

rmse_energy = np.sqrt(mean_squared_error(y_test, energy_pred))
r2_energy = r2_score(y_test, energy_pred)
accuracy_energy = r2_energy * 100

print("\n⚡ Energy Prediction (TSR)")
print("Accuracy (%):", accuracy_energy)
print("RMSE:", rmse_energy)
print("R2 Score:", r2_energy)
*/
```

## Output:
<img width="411" height="306" alt="image" src="https://github.com/user-attachments/assets/57543e66-7397-401b-9b82-905f2fedebd8" />


## Result:
Thus To write a program to predict daily temperature , PM2.5 pollution level and Energy based on environmental sensor data using Random Forest Algorithm is verified.
