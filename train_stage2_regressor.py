import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

# -----------------------------
# LOAD DATA
# -----------------------------
data = pd.read_csv("regression_dataset.csv")

# -----------------------------
# FEATURES AND TARGET
# -----------------------------
X = data.drop(columns=["image", "bilirubin"])
y = data["bilirubin"]

# -----------------------------
# TRAIN TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# MODEL
# -----------------------------
model = XGBRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X_train, y_train)

# -----------------------------
# PREDICTIONS
# -----------------------------
pred = model.predict(X_test)

# -----------------------------
# METRICS
# -----------------------------
mae = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))

print("\nRegression Results")
print("-------------------")
print("MAE :", round(mae,2))
print("RMSE:", round(rmse,2))

# -----------------------------
# SHOW SAMPLE PREDICTIONS
# -----------------------------
results = pd.DataFrame({
    "Actual": y_test,
    "Predicted": pred
})

print("\nSample Predictions")
print(results.head(10))