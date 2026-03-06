import cv2
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


# -----------------------------
# TRAIN MODEL (quick training)
# -----------------------------

df = pd.read_csv("final_dataset.csv")

X = df.drop(columns=["image", "label", "yellow_index"])
y = df["label"]

scaler = StandardScaler()
X = scaler.fit_transform(X)

model = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.2,
    scale_pos_weight=4,
    eval_metric="logloss",
    random_state=42
)

model.fit(X, y)


# -----------------------------
# SCLERA EXTRACTION
# -----------------------------

def extract_sclera(img):

    img = cv2.resize(img,(256,256))

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower = np.array([0,10,160])
    upper = np.array([40,120,255])

    mask = cv2.inRange(hsv, lower, upper)

    kernel = np.ones((5,5), np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.medianBlur(mask, 7)

    contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None

    largest = max(contours, key=cv2.contourArea)

    clean_mask = np.zeros_like(mask)
    cv2.drawContours(clean_mask,[largest],-1,255,-1)

    sclera = cv2.bitwise_and(img,img,mask=clean_mask)

    return sclera


# -----------------------------
# FEATURE EXTRACTION
# -----------------------------

def extract_features(img):

    mask = np.sum(img,axis=2) > 0
    pixels = img[mask]

    mean_r = np.mean(pixels[:,2])
    mean_g = np.mean(pixels[:,1])
    mean_b = np.mean(pixels[:,0])

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab_pixels = lab[mask]

    mean_l = np.mean(lab_pixels[:,0])
    mean_a = np.mean(lab_pixels[:,1])
    mean_b_lab = np.mean(lab_pixels[:,2])

    rg_ratio = mean_r/(mean_g+1e-6)
    rb_ratio = mean_r/(mean_b+1e-6)
    bg_ratio = mean_b/(mean_g+1e-6)

    features = [
        mean_r, mean_g, mean_b,
        mean_l, mean_a, mean_b_lab,
        rg_ratio, rb_ratio, bg_ratio
    ]

    return np.array(features).reshape(1,-1)


# -----------------------------
# TEST IMAGE
# -----------------------------

image_path = "test_eye.jpg"

img = cv2.imread(image_path)

if img is None:
    print("Image not found")
    exit()

sclera = extract_sclera(img)

if sclera is None:
    print("Sclera not detected")
    exit()

features = extract_features(sclera)

features = scaler.transform(features)

prob = model.predict_proba(features)[0][1]

print("\nPrediction probability:", prob)

if prob > 0.5:
    print("Prediction: Jaundice detected")
else:
    print("Prediction: Likely normal")