import pandas as pd

# Load features
features = pd.read_csv("features_improved.csv")

# Load labels
labels = pd.read_csv("master_labels_final.csv")

# Merge
final_df = pd.merge(features, labels, on="image")

# Save final dataset
final_df.to_csv("final_dataset.csv", index=False)

print("Final dataset created.")
print("Total samples:", len(final_df))
print("\nClass distribution:")
print(final_df["label"].value_counts())