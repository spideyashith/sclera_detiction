import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score
from sklearn.preprocessing import StandardScaler

from xgboost import XGBClassifier


# ----------------------------
# LOAD DATA
# ----------------------------

df = pd.read_csv("final_dataset.csv")

X = df.drop(columns=["image", "label"])
y = df["label"]


# ----------------------------
# SCALE FEATURES
# ----------------------------

scaler = StandardScaler()
X = scaler.fit_transform(X)


# ----------------------------
# CROSS VALIDATION
# ----------------------------

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

acc_scores = []
recall_scores = []
auc_scores = []


for train_idx, test_idx in kf.split(X, y):

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss"
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    acc_scores.append(accuracy_score(y_test, preds))
    recall_scores.append(recall_score(y_test, preds))
    auc_scores.append(roc_auc_score(y_test, probs))


print("\nCross Validation Results")
print("-------------------------")

print("Accuracy :", np.mean(acc_scores))
print("Recall   :", np.mean(recall_scores))
print("ROC-AUC  :", np.mean(auc_scores))