import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib

# -----------------------------
# LOAD DATASET
# -----------------------------
df = pd.read_csv("features_dataset.csv")

df["label"] = (df["bilirubin"] > 2).astype(int)

features = [
"mean_r","mean_g","mean_b",
"mean_h","mean_s","mean_v",
"mean_l","mean_a","mean_b_lab",
"yellow_index"
]

X = df[features]
y = df["label"]

print("Dataset size:", len(df))
print("Class distribution:")
print(y.value_counts())

# -----------------------------
# NORMALIZE FEATURES
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# save scaler
joblib.dump(scaler,"feature_scaler.pkl")

# -----------------------------
# TRAIN TEST SPLIT
# -----------------------------
X_train,X_test,y_train,y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------------
# CLASSIFIER
# -----------------------------
model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train,y_train)

# -----------------------------
# EVALUATION
# -----------------------------
pred = model.predict(X_test)

print("\nConfusion Matrix")
print(confusion_matrix(y_test,pred))

print("\nClassification Report")
print(classification_report(y_test,pred))

# -----------------------------
# SAVE MODEL
# -----------------------------
joblib.dump(model,"jaundice_classifier.pkl")

print("\nModel saved as jaundice_classifier.pkl")