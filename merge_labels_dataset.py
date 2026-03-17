import os
import pandas as pd

# -------- FILE PATHS --------

JAUNDICE_FILE = "labels.csv"
NORMAL_FOLDER = "images/normal"

OUTPUT_FILE = "merged_labels.csv"


# -------- LOAD JAUNDICE DATA --------

df_jaundice = pd.read_csv(JAUNDICE_FILE)

# add gender column
df_jaundice["gender"] = ""

print("Loaded jaundice samples:", len(df_jaundice))


# -------- LOAD NORMAL IMAGES --------

normal_rows = []

for file in os.listdir(NORMAL_FOLDER):

    if file.lower().endswith((".jpg",".jpeg",".png")):

        normal_rows.append({
            "image": file,
            "bilirubin": 1.0,
            "gender": ""
        })


df_normal = pd.DataFrame(normal_rows)

print("Loaded normal images:", len(df_normal))


# -------- MERGE DATA --------

df_final = pd.concat([df_jaundice, df_normal], ignore_index=True)


# -------- SAVE DATASET --------

df_final.to_csv(OUTPUT_FILE, index=False)


print("\nMerged dataset saved as:", OUTPUT_FILE)
print("Total samples:", len(df_final))