import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

df = pd.read_csv("final_dataset.csv")

X = df.drop(columns=["image", "bilirubin", "jaundice"])
y = df["jaundice"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Balance training data
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

print("After SMOTE:", np.bincount(y_train))

model = XGBClassifier(
    max_depth=4,
    n_estimators=300,
    learning_rate=0.05,
    scale_pos_weight=1,
    random_state=42,
    use_label_encoder=False,
    eval_metric="logloss"
)

model.fit(X_train, y_train)

# Use tuned threshold (replace with your value)
threshold = 0.35

probs = model.predict_proba(X_test)[:, 1]
y_pred = (probs > threshold).astype(int)

print("\n===== FINAL STAGE 1 RESULTS =====")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, probs))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))