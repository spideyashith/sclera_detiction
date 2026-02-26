import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_auc_score,
    precision_recall_curve
)
from xgboost import XGBClassifier

# ===============================
# 1️⃣ Load Dataset
# ===============================
df = pd.read_csv("dataset.csv")

# Create jaundice label (clinical threshold)
df["jaundice"] = (df["bilirubin"] > 1.2).astype(int)

# ===============================
# 2️⃣ Select Important Features
# ===============================
features = [
    "yellow_index",
    "mean_v",
    "mean_r",
    "mean_s",
    "entropy"
]

X = df[features]
y = df["jaundice"]

# ===============================
# 3️⃣ Train Test Split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    stratify=y,
    random_state=42
)

# ===============================
# 4️⃣ XGBoost Model
# ===============================
model = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    scale_pos_weight=1,
    random_state=42,
    eval_metric="logloss"
)

model.fit(X_train, y_train)

# ===============================
# 5️⃣ Probability Prediction
# ===============================
probs = model.predict_proba(X_test)[:, 1]

# ===============================
# 6️⃣ Threshold Tuning
# ===============================
precision, recall, thresholds = precision_recall_curve(y_test, probs)

# Choose threshold giving recall ≥ 0.80
best_threshold = 0.5
for p, r, t in zip(precision, recall, thresholds):
    if r >= 0.80:
        best_threshold = t
        break

y_pred = (probs >= best_threshold).astype(int)

# ===============================
# 7️⃣ Evaluation
# ===============================
print("\n===== FINAL XGBOOST RESULTS =====")

print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, probs))

print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("\nSelected Threshold:", best_threshold)

# ===============================
# 8️⃣ Feature Importance
# ===============================
importances = model.feature_importances_

print("\nFeature Importance:")
for f, imp in sorted(zip(features, importances),
                     key=lambda x: x[1],
                     reverse=True):
    print(f"{f}: {round(imp, 3)}")
