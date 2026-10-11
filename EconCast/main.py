# Build a one-month-ahead inflation forecast from the historical CSV file.
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Read the CSV stored beside this script, so it works from any launch folder.
data = pd.read_csv(Path(__file__).with_name("inflation.csv"))
values = data["Inflation Rate (YoY)"].to_numpy(dtype=float)

# Use the previous 24 months as the input for each next-month prediction.
window_size = 24
if len(values) <= window_size:
    raise ValueError(f"Need more than {window_size} rows; found {len(values)}.")

# X holds each 24-month input window. y holds the matching following-month value.
X = np.array([
    values[i : i + window_size]
    for i in range(len(values) - window_size)
])
y = values[window_size:]

# Keep the data in time order: train on the first 80%, test on the last 20%.
split = int(len(X) * 0.8)
if split == 0 or split == len(X):
    raise ValueError("Not enough examples for both training and testing.")

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Train a linear regression algorithm to learn from the earlier examples.
model = LinearRegression()
model.fit(X_train, y_train)

# Predict each test month's inflation using its previous 24 months.
predictions = model.predict(X_test)

# Baseline comparison: assume next month's rate will equal the latest month's rate.
# This simple forecast is copied from the final value in each 24-month input window.
baseline_predictions = X_test[:, -1]

# Report average absolute error and root mean squared error; lower is better.
def report(name, actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    rmse = mean_squared_error(actual, predicted) ** 0.5
    print(f"{name}: MAE={mae:.3f}, RMSE={rmse:.3f}")

report("Regression", y_test, predictions)
report("Baseline", y_test, baseline_predictions)

# Match each test result to its year and month, then display actual and predicted values.
test_start = window_size + split
comparison = data.loc[test_start:, ["Year", "Month"]].copy()
comparison["Actual"] = y_test
comparison["Regression Prediction"] = predictions
comparison["Baseline Prediction"] = baseline_predictions
print("\nTest-set predictions:")
print(comparison.to_string(index=False, formatters={
    "Actual": "{:.2f}".format,
    "Regression Prediction": "{:.2f}".format,
    "Baseline Prediction": "{:.2f}".format,
}))
