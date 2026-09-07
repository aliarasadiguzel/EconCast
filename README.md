EconCast is a simple machine learning project that predicts the next month's inflation rate using historical inflation data.

How It Works

EconCast uses Linear Regression with a 24-month input window.

Input: Previous 24 months of inflation rates Target: The following month's inflation rate Model: Linear Regression Train/Test split: 80% / 20% Shuffle: Disabled to preserve the chronological order of the data Results

The first version achieved:

MAE: 0.32 RMSE: 0.40

These values represent the model's average prediction error on the test data.

Technologies Python Pandas Scikit-learn Project Status

Version 1.0 — Baseline

This is an early machine learning project focused on learning the fundamentals of supervised learning and linear regression.

Future versions may experiment with additional features and different approaches while keeping the project focused on regression.

Disclaimer

EconCast is an educational project and is not intended for making financial or economic decisions.
