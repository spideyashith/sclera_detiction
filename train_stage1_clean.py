import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler

from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier


# -----------------------------
# LOAD DATA
# -----------------------------

df = pd.read_csv("final_dataset.csv")

# Remove useless columns
X = df.drop(columns=["image", "label", "yellow_index"])
y = df["label"]


# -----------------------------
# TRAIN TEST SPLIT
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)


# -----------------------------
# SCALE FEATURES
# -----------------------------

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# -----------------------------
# HANDLE CLASS IMBALANCE
# -----------------------------

print("Before SMOTE:", np.bincount(y_train))

smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

print("After SMOTE:", np.bincount(y_train))


# -----------------------------
# TRAIN MODEL
# -----------------------------

model = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.2,
    scale_pos_weight=4,
    eval_metric="logloss",
    random_state=42
)

model.fit(X_train, y_train)


# -----------------------------
# PREDICTIONS
# -----------------------------

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]


# -----------------------------
# RESULTS
# -----------------------------

print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report")
print(classification_report(y_test, y_pred))

print("\nROC-AUC:", roc_auc_score(y_test, y_prob))