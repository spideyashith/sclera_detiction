import pandas as pd
import xgboost as xgb
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score

# Load dataset
df = pd.read_csv("final_dataset.csv")

# Features and labels
X = df.drop(columns=["image","label"])
y = df["label"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train model
model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    random_state=42
)

model.fit(X_train, y_train)

# Predicted probabilities
probs = model.predict_proba(X_test)[:,1]

# Try different thresholds
best_threshold = 0
best_recall = 0

print("\nThreshold Testing\n")

for t in np.arange(0.2,0.9,0.05):

    preds = (probs >= t).astype(int)

    recall = recall_score(y_test, preds)
    precision = precision_score(y_test, preds)

    print(f"Threshold {t:.2f}  Recall {recall:.3f}  Precision {precision:.3f}")

    if recall > best_recall:
        best_recall = recall
        best_threshold = t

print("\nBest Threshold:", best_threshold)
print("Best Recall:", best_recall)