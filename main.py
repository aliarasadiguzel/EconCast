import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
data = pd.read_csv("inflation.csv")

input_windows = []
target_window = []

for i in range(len(data) - 24):
    window = data["Inflation Rate (YoY)"][i:i+24].tolist()
    target = data["Inflation Rate (YoY)"][i+24]

    input_windows.append(window)
    target_window.append(target)

X_train, X_test, y_train, y_test = train_test_split(
    input_windows,
    target_window,
    test_size=0.2,
    shuffle=False
)

model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

for prediction, actual in zip(predictions, y_test):
    print(f"Predicted: {prediction:.2f} | Actual: {actual:.2f}")

mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)

print("MAE:", round(mae, 2))
print("RMSE:", round(mse ** 0.5, 2))