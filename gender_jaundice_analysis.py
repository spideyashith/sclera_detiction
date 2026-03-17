import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("merged_labels.csv")

# Create jaundice label
df["jaundice"] = (df["bilirubin"] > 2).astype(int)

print("Total samples:", len(df))

# Gender count
print("\nGender distribution")
print(df["gender"].value_counts())

# Jaundice comparison
table = pd.crosstab(df["gender"], df["jaundice"])
table.columns = ["Normal", "Jaundice"]

print("\nJaundice vs Normal by Gender")
print(table)

# Percentage
percentage = df.groupby("gender")["jaundice"].mean() * 100

print("\nPercentage with Jaundice")
print(percentage)

# Plot
percentage.plot(kind="bar", color=["blue","orange"])

plt.title("Jaundice Percentage by Gender")
plt.ylabel("Percentage")
plt.xlabel("Gender")

plt.tight_layout()

plt.savefig("gender_jaundice_comparison.png")

plt.show()

