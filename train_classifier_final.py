# train_classifier_final.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)
from sklearn.preprocessing import StandardScaler

# ----------------------------
# 1. Load Dataset
# ----------------------------
df = pd.read_csv("dataset.csv")

# Create Jaundice label
df["jaundice"] = (df["bilirubin"] > 1.2).astype(int)

# ----------------------------
# 2. Select Features
# ----------------------------
features = [
    "mean_r",
    "mean_g",
    "mean_b",
    "mean_h",
    "mean_s",
    "mean_v",
    "entropy",
    "yellow_index"
]

X = df[features]
y = df["jaundice"]

# ----------------------------
# 3. Feature Scaling
# ----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ----------------------------
# 4. Stratified Train-Test Split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.25,
    stratify=y,
    random_state=42
)

# ----------------------------
# 5. Balanced RandomForest
# ----------------------------
model = RandomForestClassifier(
    n_estimators=400,
    max_depth=None,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)

# ----------------------------
# 6. Prediction with probability threshold tuning
# ----------------------------
probs = model.predict_proba(X_test)[:, 1]

# Adjust threshold (important!)
threshold = 0.45
y_pred = (probs >= threshold).astype(int)

# ----------------------------
# 7. Evaluation
# ----------------------------
print("\n===== FINAL CLASSIFICATION RESULTS =====")

print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, probs))

print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("\nJaundice Detection Recall:",
      classification_report(y_test, y_pred, output_dict=True)["1"]["recall"])

# ----------------------------
# 8. Feature Importance
# ----------------------------
importances = model.feature_importances_
print("\nFeature Importance:")
for name, imp in sorted(zip(features, importances), key=lambda x: -x[1]):
    print(f"{name}: {imp:.3f}")
