import pandas as pd


# Features from augmented images
features = pd.read_csv("features_aug.csv")

# Patient-level bilirubin
patients = pd.read_csv("patient_summary.csv")


# Extract patient ID from image name
features["patient"] = features["image"].str.extract(r"(AJ\d+)")

# Merge
final = features.merge(
    patients[["patient","bilirubin"]],
    on="patient",
    how="inner"
)

# Remove helper column
final = final.drop(columns=["patient"])


final.to_csv("dataset.csv", index=False)

print("Saved: dataset.csv")
print("Total rows:", len(final))
