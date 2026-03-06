import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load dataset
df = pd.read_csv("features_final.csv")

# Features used for ML
features = [
    "mean_r",
    "mean_g",
    "mean_b",
    "mean_l",
    "mean_a",
    "mean_b_lab",
    "rg_ratio",
    "rb_ratio",
    "bg_ratio",
    "yellow_index"
]

X = df[features]
y = df["label"]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Plot
plt.figure(figsize=(7,6))

for label in [0,1]:
    subset = X_pca[y == label]
    
    if label == 0:
        plt.scatter(subset[:,0], subset[:,1], label="Normal", alpha=0.7)
    else:
        plt.scatter(subset[:,0], subset[:,1], label="Jaundice", alpha=0.7)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA Feature Distribution")

plt.legend()
plt.grid(True)

plt.show()