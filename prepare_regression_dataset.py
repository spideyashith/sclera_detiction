import pandas as pd

# ----------------------------
# LOAD DATA
# ----------------------------
features = pd.read_csv("features_improved.csv")
labels = pd.read_csv("labels.csv")

# ----------------------------
# MERGE FEATURES + BILIRUBIN
# ----------------------------
data = pd.merge(features, labels, on="image")

# ----------------------------
# REMOVE NORMAL SAMPLES
# bilirubin NA means normal
# ----------------------------
data = data.dropna(subset=["bilirubin"])

# ----------------------------
# SAVE DATASET
# ----------------------------
data.to_csv("regression_dataset.csv", index=False)

print("Regression dataset created")
print("Total samples:", len(data))
print(data.head())