import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("dataset.csv")

# Create jaundice label (medical threshold)
df["jaundice"] = (df["bilirubin"] > 1.2).astype(int)

# Features
features = ["mean_r","mean_g","mean_b","mean_h","mean_s","mean_v","entropy","yellow_index"]
X = df[features]
y = df["jaundice"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ===== BALANCE DATASET USING SMOTE =====
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X_scaled, y)

print("Before balance:", np.bincount(y))
print("After balance:", np.bincount(y_balanced))

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_balanced, y_balanced,
    test_size=0.25,
    random_state=42,
    stratify=y_balanced
)

# Train model
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=6,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Metrics
print("\n===== CLASSIFICATION RESULTS =====")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Important medical metric
recall = recall_score(y_test, y_pred)
print("\nJaundice Detection Recall:", recall)

# Feature importance
importance = model.feature_importances_
for name, score in zip(features, importance):
    print(f"{name}: {score:.3f}")


