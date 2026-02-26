import pandas as pd

df = pd.read_csv("dataset.csv")

# Create binary target
df["jaundice"] = (df["bilirubin"] > 1.2).astype(int)

# Save clean dataset
df.to_csv("final_dataset.csv", index=False)

print("Saved: final_dataset.csv")
print(df["jaundice"].value_counts())
