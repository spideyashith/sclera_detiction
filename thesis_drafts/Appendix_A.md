# APPENDIX A: SOURCE CODE FOR MODEL TRAINING

In this appendix, we present the core Python implementation for training the two-stage predictive models used in this research.

## A.1 Stage 1: Jaundice Classification Training
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# Load engineered color features
df = pd.read_csv("features_final.csv")
X = df.drop(columns=["image", "label"])
y = df["label"]

# Perform stratified train/test split (80-20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Address class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Initialize and train binary XGBoost Classifier
classifier = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    random_state=42
)
classifier.fit(X_train_res, y_train_res)

# Evaluate on unseen test data
y_pred = classifier.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))
```

## A.2 Stage 2: Bilirubin Level Regression Training
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
import joblib

# Load dataset and isolate positive cases for regression training
df = pd.read_csv("features_dataset.csv")
df = df[df["bilirubin"] > 2.0]

features = ["mean_r", "mean_g", "mean_b", "mean_h", "mean_s", "mean_v", 
            "mean_l", "mean_a", "mean_b_lab", "yellow_index"]
X = df[features]
y = df["bilirubin"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize and optimize XGBoost Regressor
regressor = xgb.XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    random_state=42
)
regressor.fit(X_train, y_train)

# Calculate Mean Absolute Error (MAE)
predictions = regressor.predict(X_test)
print("Mean Absolute Error (mg/dL):", mean_absolute_error(y_test, predictions))

# Export trained model for production use
joblib.dump(regressor, "bilirubin_regressor.pkl")
```
