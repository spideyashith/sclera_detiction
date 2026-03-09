import cv2
import torch
import numpy as np
import joblib
import pandas as pd
import segmentation_models_pytorch as smp

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMG_SIZE = 256

MODEL_PATH = "sclera_segmentation_model.pth"

# Load ML models
classifier = joblib.load("jaundice_classifier.pkl")
regressor = joblib.load("bilirubin_regressor.pkl")

# -----------------------------
# LOAD SEGMENTATION MODEL
# -----------------------------
seg_model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=3,
    classes=1
)

seg_model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
seg_model.to(DEVICE)
seg_model.eval()

# -----------------------------
# GRAY WORLD COLOR NORMALIZATION
# -----------------------------
def gray_world_normalization(img):

    img = img.astype(np.float32)

    avg_b = np.mean(img[:,:,0])
    avg_g = np.mean(img[:,:,1])
    avg_r = np.mean(img[:,:,2])

    avg_gray = (avg_b + avg_g + avg_r) / 3

    img[:,:,0] = img[:,:,0] * (avg_gray / avg_b)
    img[:,:,1] = img[:,:,1] * (avg_gray / avg_g)
    img[:,:,2] = img[:,:,2] * (avg_gray / avg_r)

    img = np.clip(img,0,255)

    return img.astype(np.uint8)

# -----------------------------
# SEGMENT SCLERA
# -----------------------------
def segment_sclera(image):

    img = cv2.resize(image,(IMG_SIZE,IMG_SIZE))
    img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    img_norm = img_rgb/255.0
    img_norm = np.transpose(img_norm,(2,0,1))
    img_norm = np.expand_dims(img_norm,0)

    tensor = torch.tensor(img_norm).float().to(DEVICE)

    with torch.no_grad():
        pred = seg_model(tensor)

    pred = torch.sigmoid(pred).cpu().numpy()[0][0]

    mask = (pred > 0.5).astype(np.uint8) * 255
    mask = cv2.resize(mask,(image.shape[1],image.shape[0]))

    return mask

# -----------------------------
# FEATURE EXTRACTION
# -----------------------------
def extract_features(image,mask):

    sclera = cv2.bitwise_and(image,image,mask=mask)

    pixels = sclera[mask>0]

    if len(pixels)==0:
        return None

    mean_b = np.mean(pixels[:,0])
    mean_g = np.mean(pixels[:,1])
    mean_r = np.mean(pixels[:,2])

    hsv = cv2.cvtColor(sclera,cv2.COLOR_BGR2HSV)
    hsv_pixels = hsv[mask>0]

    mean_h = np.mean(hsv_pixels[:,0])
    mean_s = np.mean(hsv_pixels[:,1])
    mean_v = np.mean(hsv_pixels[:,2])

    lab = cv2.cvtColor(sclera,cv2.COLOR_BGR2LAB)
    lab_pixels = lab[mask>0]

    mean_l = np.mean(lab_pixels[:,0])
    mean_a = np.mean(lab_pixels[:,1])
    mean_b_lab = np.mean(lab_pixels[:,2])

    yellow_index = mean_r - mean_b

    features = [
        mean_r,mean_g,mean_b,
        mean_h,mean_s,mean_v,
        mean_l,mean_a,mean_b_lab,
        yellow_index
    ]

    return np.array(features).reshape(1,-1)

# -----------------------------
# VISUALIZATION
# -----------------------------
def overlay_mask(image,mask):

    overlay = image.copy()
    overlay[mask>0] = [0,255,0]

    blended = cv2.addWeighted(image,0.7,overlay,0.3,0)

    return blended

# -----------------------------
# TEST IMAGE
# -----------------------------
IMAGE_PATH = "normal_eye.jpg"

image = cv2.imread(IMAGE_PATH)

if image is None:
    print("Image not found.")
    exit()

# Apply color normalization
image = gray_world_normalization(image)

# Segment sclera
mask = segment_sclera(image)

# Extract features
features = extract_features(image,mask)

if features is None:
    print("Could not detect sclera")
    exit()

feature_names = [
"mean_r","mean_g","mean_b",
"mean_h","mean_s","mean_v",
"mean_l","mean_a","mean_b_lab",
"yellow_index"
]

features_df = pd.DataFrame(features,columns=feature_names)

# -----------------------------
# CLASSIFICATION
# -----------------------------
prob = classifier.predict_proba(features_df)[0][1]

print("Jaundice Probability:",round(prob,3))

THRESHOLD = 0.75

if prob < THRESHOLD:

    print("\nPrediction: NORMAL")
    print("Estimated Bilirubin: < 2 mg/dL")

else:

    print("\nPrediction: JAUNDICE DETECTED")

    bilirubin = regressor.predict(features_df)[0]

    print("Estimated Bilirubin:",round(bilirubin,2),"mg/dL")

# -----------------------------
# SAVE VISUALIZATION
# -----------------------------
overlay = overlay_mask(image,mask)

cv2.imwrite("predicted_mask.png",mask)
cv2.imwrite("sclera_overlay.png",overlay)

print("\nSaved:")
print("predicted_mask.png")
print("sclera_overlay.png")