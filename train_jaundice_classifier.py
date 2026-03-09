import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# load features
df = pd.read_csv("features_dataset.csv")

# create label
df["label"] = (df["bilirubin"] > 2).astype(int)

# feature columns
features = [
"mean_r","mean_g","mean_b",
"mean_h","mean_s","mean_v",
"mean_l","mean_a","mean_b_lab",
"yellow_index"
]

X = df[features]
y = df["label"]

# split dataset
X_train,X_test,y_train,y_test = train_test_split(
X,y,test_size=0.2,random_state=42,stratify=y
)

# train model
model = RandomForestClassifier(
n_estimators=200,
random_state=42
)

model.fit(X_train,y_train)

# predictions
pred = model.predict(X_test)

print("Confusion Matrix")
print(confusion_matrix(y_test,pred))

print("\nClassification Report")
print(classification_report(y_test,pred))

# save model
joblib.dump(model,"jaundice_classifier.pkl")

print("\nModel saved as jaundice_classifier.pkl")