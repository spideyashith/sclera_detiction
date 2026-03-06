import cv2
import numpy as np
import pandas as pd
import xgboost as xgb

# Load trained dataset
df = pd.read_csv("final_dataset.csv")

X = df.drop(columns=["image","label"])
y = df["label"]

# Train model again (same as stage1)
model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    random_state=42
)

model.fit(X, y)

# -------- FEATURE EXTRACTION FUNCTION --------

def extract_features(image_path):

    img = cv2.imread(image_path)

    img = cv2.resize(img,(256,256))

    mask = np.sum(img,axis=2) > 0
    pixels = img[mask]

    mean_r = np.mean(pixels[:,2])
    mean_g = np.mean(pixels[:,1])
    mean_b = np.mean(pixels[:,0])

    lab = cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
    lab_pixels = lab[mask]

    mean_l = np.mean(lab_pixels[:,0])
    mean_a = np.mean(lab_pixels[:,1])
    mean_b_lab = np.mean(lab_pixels[:,2])

    yellow_index = mean_b_lab

    rg_ratio = mean_r/mean_g
    rb_ratio = mean_r/mean_b
    bg_ratio = mean_b/mean_g

    return [[
        mean_r,mean_g,mean_b,
        mean_l,mean_a,mean_b_lab,
        yellow_index,
        rg_ratio,rb_ratio,bg_ratio
    ]]


# -------- TEST IMAGE --------

image_path = "test_eye.jpg"

features = extract_features(image_path)

prob = model.predict_proba(features)[0][1]

print("\nPrediction Probability:",prob)

# -------- RISK LEVEL --------

if prob < 0.20:
    print("Result: Likely Normal")

elif prob < 0.50:
    print("Result: Mild Risk - Recommend Test")

else:
    print("Result: High Risk of Jaundice")