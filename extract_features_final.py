import cv2
import numpy as np
import pandas as pd
import os

INPUT_FOLDER = "sclera_clean"
OUTPUT_FILE = "features_final.csv"

data = []

for root, dirs, files in os.walk(INPUT_FOLDER):

    for file in files:

        if not (file.endswith(".jpg") or file.endswith(".png") or file.endswith(".jpeg")):
            continue

        path = os.path.join(root, file)

        img = cv2.imread(path)

        if img is None:
            continue


        img = cv2.resize(img, (256,256))


        # Remove black background
        mask = np.sum(img, axis=2) > 0

        pixels = img[mask]

        if len(pixels) < 30:
            continue


        # RGB
        mean_r = np.mean(pixels[:,2])
        mean_g = np.mean(pixels[:,1])
        mean_b = np.mean(pixels[:,0])


        # LAB
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        lab_pixels = lab[mask]

        mean_l = np.mean(lab_pixels[:,0])
        mean_a = np.mean(lab_pixels[:,1])
        mean_b_lab = np.mean(lab_pixels[:,2])


        yellow_index = mean_b_lab


        # ratios
        rg_ratio = mean_r / (mean_g + 1e-6)
        rb_ratio = mean_r / (mean_b + 1e-6)
        bg_ratio = mean_b / (mean_g + 1e-6)

        yellow_ratio = mean_b_lab / (mean_l + 1e-6)


        # label from folder
        if "jaundice" in root:
            label = 1
        else:
            label = 0


        data.append([
            file,
            mean_r,
            mean_g,
            mean_b,
            mean_l,
            mean_a,
            mean_b_lab,
            yellow_index,
            rg_ratio,
            rb_ratio,
            bg_ratio,
            yellow_ratio,
            label
        ])


columns = [
    "image",
    "mean_r",
    "mean_g",
    "mean_b",
    "mean_l",
    "mean_a",
    "mean_b_lab",
    "yellow_index",
    "rg_ratio",
    "rb_ratio",
    "bg_ratio",
    "yellow_ratio",
    "label"
]

df = pd.DataFrame(data, columns=columns)

df.to_csv(OUTPUT_FILE, index=False)

print("Feature dataset created")
print("Total samples:", len(df))
print(df["label"].value_counts())