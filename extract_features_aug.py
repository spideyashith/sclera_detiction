import cv2
import numpy as np
import os
import pandas as pd
from skimage.measure import shannon_entropy


IMG_DIR = "sclera_clean_aug"
OUT_CSV = "features_aug.csv"

rows = []


for file in os.listdir(IMG_DIR):

    if not file.lower().endswith(".jpg"):
        continue

    path = os.path.join(IMG_DIR, file)

    img = cv2.imread(path)

    if img is None:
        continue


    img = cv2.resize(img, (128,128))

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    # RGB
    mean_r = np.mean(rgb[:,:,0])
    mean_g = np.mean(rgb[:,:,1])
    mean_b = np.mean(rgb[:,:,2])

    # HSV
    mean_h = np.mean(hsv[:,:,0])
    mean_s = np.mean(hsv[:,:,1])
    mean_v = np.mean(hsv[:,:,2])

    # Entropy
    ent = shannon_entropy(gray)

    # Yellow index
    yellow = (mean_r + mean_g) / (2*mean_b + 1e-5)


    rows.append([
        file,
        mean_r, mean_g, mean_b,
        mean_h, mean_s, mean_v,
        ent, yellow
    ])


df = pd.DataFrame(rows, columns=[
    "image",
    "mean_r","mean_g","mean_b",
    "mean_h","mean_s","mean_v",
    "entropy","yellow_index"
])

df.to_csv(OUT_CSV, index=False)

print("Saved:", OUT_CSV)
print("Total samples:", len(df))
