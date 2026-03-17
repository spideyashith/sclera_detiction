import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch
import joblib
import segmentation_models_pytorch as smp

# -----------------------------
# DEVICE SETUP
# -----------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 256

# -----------------------------
# LOAD MODELS
# -----------------------------

MODEL_PATH = "sclera_segmentation_model.pth"

classifier = joblib.load("jaundice_classifier.pkl")
regressor = joblib.load("bilirubin_regressor.pkl")
scaler = joblib.load("feature_scaler.pkl")

# segmentation model
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
# SEGMENT SCLERA
# -----------------------------

def segment_sclera(image):

    img = cv2.resize(image,(IMG_SIZE,IMG_SIZE))
    img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    img_norm = img_rgb / 255.0
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
# STREAMLIT UI
# -----------------------------

st.title("AI-Based Jaundice Detection from Eye Image")

st.write("Upload an eye image to detect jaundice and estimate bilirubin level.")

uploaded_file = st.file_uploader("Upload Eye Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    image = np.array(image)

    st.subheader("Uploaded Image")
    st.image(image, use_column_width=True)

    if st.button("Run Detection"):

        mask = segment_sclera(image)

        st.subheader("Predicted Sclera Mask")
        st.image(mask, use_column_width=True)

        sclera = cv2.bitwise_and(image,image,mask=mask)

        st.subheader("Extracted Sclera Region")
        st.image(sclera, use_column_width=True)

        features = extract_features(image,mask)

        if features is None:
            st.error("Could not detect sclera properly.")
            st.stop()

        features = scaler.transform(features)

        prob = classifier.predict_proba(features)[0][1]

        st.subheader("Prediction Results")

        st.write("Jaundice Probability:", round(prob,3))

        if prob < 0.8:

            st.success("Prediction: NORMAL")
            st.write("Estimated Bilirubin: < 2 mg/dL")

        else:

            bilirubin = regressor.predict(features)[0]

            st.error("Prediction: JAUNDICE DETECTED")
            st.write("Estimated Bilirubin:", round(bilirubin,2),"mg/dL")

            st.subheader("Suggested Medical Advice")

            st.write("""
Possible medications commonly used in jaundice treatment:

• Ursodeoxycholic acid  
• Silymarin  
• Vitamin B complex  

⚠ This system is not a medical diagnosis. Please consult a doctor.
""")