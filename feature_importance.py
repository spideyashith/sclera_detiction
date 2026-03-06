import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("final_dataset.csv")

# Features
X = df.drop(columns=["image", "label"])

# Target
y = df["label"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train XGBoost model
model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    random_state=42
)

model.fit(X_train, y_train)

# Get feature importance
importance = model.feature_importances_
features = X.columns

# Create dataframe
imp_df = pd.DataFrame({
    "feature": features,
    "importance": importance
}).sort_values(by="importance", ascending=False)

print("\nFeature Importance Ranking:\n")
print(imp_df)

# Plot
plt.figure(figsize=(8,5))
plt.barh(imp_df["feature"], imp_df["importance"])
plt.gca().invert_yaxis()
plt.title("Feature Importance (XGBoost)")
plt.xlabel("Importance")
plt.tight_layout()

plt.savefig("feature_importance.png")
plt.show()