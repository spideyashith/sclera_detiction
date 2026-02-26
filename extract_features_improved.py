import cv2
import numpy as np
import pandas as pd
import os

input_folder = "sclera_clean"
output_csv = "features_improved.csv"

data = []

for file in os.listdir(input_folder):
    if file.endswith(".jpg") or file.endswith(".png"):
        path = os.path.join(input_folder, file)

        img = cv2.imread(path)
        if img is None:
            continue

        # Remove black background
        mask = np.sum(img, axis=2) > 0
        pixels = img[mask]

        if len(pixels) == 0:
            continue

        mean_r = np.mean(pixels[:, 2])
        mean_g = np.mean(pixels[:, 1])
        mean_b = np.mean(pixels[:, 0])

        # Convert to LAB
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        lab_pixels = lab[mask]

        mean_l = np.mean(lab_pixels[:, 0])
        mean_a = np.mean(lab_pixels[:, 1])
        mean_b_lab = np.mean(lab_pixels[:, 2])   # THIS IS IMPORTANT

        # Improved Yellow Index (LAB-based)
        yellow_index = mean_b_lab

        data.append([
            file, mean_r, mean_g, mean_b,
            mean_l, mean_a, mean_b_lab,
            yellow_index
        ])

columns = [
    "image", "mean_r", "mean_g", "mean_b",
    "mean_l", "mean_a", "mean_b_lab",
    "yellow_index"
]

df = pd.DataFrame(data, columns=columns)
df.to_csv(output_csv, index=False)

print("Saved:", output_csv)
print("Total samples:", len(df))