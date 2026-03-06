import cv2
import numpy as np
import pandas as pd
import os

input_folder = "sclera_clean"
output_csv = "features_improved.csv"

data = []

for class_name in ["jaundice", "normal"]:

    class_folder = os.path.join(input_folder, class_name)

    for file in os.listdir(class_folder):

        if not file.lower().endswith((".jpg",".jpeg",".png")):
            continue

        path = os.path.join(class_folder, file)

        img = cv2.imread(path)

        if img is None:
            continue

        # Remove black background
        mask = np.sum(img, axis=2) > 0
        pixels = img[mask]

        if len(pixels) == 0:
            continue

        mean_r = np.mean(pixels[:,2])
        mean_g = np.mean(pixels[:,1])
        mean_b = np.mean(pixels[:,0])


        rg_ratio = mean_r / (mean_g + 1e-6)
        rb_ratio = mean_r / (mean_b + 1e-6)
        bg_ratio = mean_b / (mean_g + 1e-6)

        # LAB color space
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        lab_pixels = lab[mask]

        mean_l = np.mean(lab_pixels[:,0])
        mean_a = np.mean(lab_pixels[:,1])
        mean_b_lab = np.mean(lab_pixels[:,2])

        yellow_index = mean_b_lab

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
            bg_ratio
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
    "bg_ratio"
]

df = pd.DataFrame(data, columns=columns)

df.to_csv(output_csv, index=False)

print("Saved:", output_csv)
print("Total samples:", len(df))