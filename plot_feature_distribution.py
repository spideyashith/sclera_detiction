import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("features_final.csv")

# Separate classes
jaundice = df[df["label"] == 1]
normal = df[df["label"] == 0]

features = [
    "mean_r",
    "mean_g",
    "mean_b",
    "mean_b_lab",
    "yellow_index"
]

for feature in features:

    plt.figure(figsize=(6,4))

    sns.kdeplot(jaundice[feature], label="Jaundice", fill=True)
    sns.kdeplot(normal[feature], label="Normal", fill=True)

    plt.title(f"{feature} Distribution")
    plt.xlabel(feature)
    plt.ylabel("Density")
    plt.legend()

    plt.tight_layout()

    plt.savefig(f"{feature}_distribution.png")

    plt.show()