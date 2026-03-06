import os
import pandas as pd

base_folder = "images"

data = []

for label_name in ["jaundice", "normal"]:

    folder = os.path.join(base_folder, label_name)

    for file in os.listdir(folder):

        if file.lower().endswith((".jpg",".jpeg",".png")):

            if label_name == "jaundice":
                label = 1
            else:
                label = 0

            data.append([file,label])

df = pd.DataFrame(data, columns=["image","label"])

df.to_csv("master_labels_final.csv",index=False)

print("Label file recreated")
print(df["label"].value_counts())