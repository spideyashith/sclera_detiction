import cv2
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor


# -----------------------------
# LOAD CLASSIFICATION DATA
# -----------------------------

df = pd.read_csv("final_dataset.csv")

X = df.drop(columns=["image","label"])
y = df["label"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# -----------------------------
# TRAIN CLASSIFIER
# -----------------------------

clf = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42
)

clf.fit(X_scaled,y)


# -----------------------------
# LOAD REGRESSION DATA
# -----------------------------

reg_df = pd.read_csv("regression_dataset.csv")

Xr = reg_df.drop(columns=["image","bilirubin"])
yr = reg_df["bilirubin"]


# -----------------------------
# TRAIN REGRESSION MODEL
# -----------------------------

reg = XGBRegressor(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    random_state=42
)

reg.fit(Xr,yr)


# -----------------------------
# SCLERA EXTRACTION
# -----------------------------

def extract_sclera(img):

    img = cv2.resize(img,(256,256))

    # LAB brightness normalization
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    l,a,b = cv2.split(lab)

    l = cv2.equalizeHist(l)

    lab = cv2.merge((l,a,b))

    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

    lower = np.array([0,5,140])
    upper = np.array([50,140,255])

    mask = cv2.inRange(hsv,lower,upper)

    kernel = np.ones((7,7),np.uint8)

    mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)
    mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)
    mask = cv2.medianBlur(mask,7)

    contours,_ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours)==0:
        return None

    contours = sorted(contours,key=cv2.contourArea,reverse=True)

    clean_mask = np.zeros_like(mask)

    for c in contours[:2]:
        cv2.drawContours(clean_mask,[c],-1,255,-1)

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

    lab = cv2.cvtColor(img,cv2.COLOR_BGR2LAB)

    lab_pixels = lab[mask]

    mean_l = np.mean(lab_pixels[:,0])
    mean_a = np.mean(lab_pixels[:,1])
    mean_b_lab = np.mean(lab_pixels[:,2])

    # yellow index
    yellow_index = mean_b_lab

    rg_ratio = mean_r/(mean_g+1e-6)
    rb_ratio = mean_r/(mean_b+1e-6)
    bg_ratio = mean_b/(mean_g+1e-6)

    features = [
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
    ]

    return np.array(features).reshape(1,-1)


# -----------------------------
# TEST IMAGE
# -----------------------------

image_path = "nor_eye.jpg"

img = cv2.imread(image_path)

if img is None:
    print("Image not found")
    exit()


sclera = extract_sclera(img)

if sclera is None:
    print("Sclera not detected")
    exit()


features = extract_features(sclera)

features_scaled = scaler.transform(features)


# -----------------------------
# STAGE 1: JAUNDICE DETECTION
# -----------------------------

prob = clf.predict_proba(features_scaled)[0][1]

print("\nJaundice Probability:",round(prob,3))


if prob > 0.5:

    print("Prediction: Jaundice detected")

    bilirubin = reg.predict(features)[0]

    print("Estimated Bilirubin:",round(bilirubin,2),"mg/dL")

else:

    print("Prediction: Normal eye")