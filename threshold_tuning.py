import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from xgboost import XGBClassifier

df = pd.read_csv("final_dataset.csv")

X = df.drop(columns=["image", "bilirubin", "jaundice"])
y = df["jaundice"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

model = XGBClassifier(
    scale_pos_weight=1,
    random_state=42,
    use_label_encoder=False,
    eval_metric="logloss"
)

model.fit(X_train, y_train)

probs = model.predict_proba(X_test)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, probs)

# Youden’s Index
best_index = np.argmax(tpr - fpr)
best_threshold = thresholds[best_index]

print("Best Threshold:", best_threshold)