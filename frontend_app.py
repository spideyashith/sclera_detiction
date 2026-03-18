import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch
import joblib
import segmentation_models_pytorch as smp

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(layout="wide")

# -----------------------------
# DEVICE
# -----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 256

# -----------------------------
# LOAD MODELS
# -----------------------------
MODEL_PATH = "sclera_segmentation_model.pth"

@st.cache_resource
def load_models():
    """Caches models to prevent reloading on every UI interaction."""
    classifier = joblib.load("jaundice_classifier.pkl")
    regressor = joblib.load("bilirubin_regressor.pkl")
    scaler = joblib.load("feature_scaler.pkl")

    seg_model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1
    )

    seg_model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    seg_model.to(DEVICE)
    seg_model.eval()
    
    return classifier, regressor, scaler, seg_model

try:
    classifier, regressor, scaler, seg_model = load_models()
except Exception as e:
    st.error(f"Failed to load models. Ensure all .pkl and .pth files are in the directory. Error: {e}")
    st.stop()

# -----------------------------
# SEGMENT SCLERA
# -----------------------------
def segment_sclera(image):
    img = cv2.resize(image, (IMG_SIZE, IMG_SIZE))  
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_norm = img_rgb / 255.0  
    img_norm = np.transpose(img_norm, (2, 0, 1))  
    img_norm = np.expand_dims(img_norm, 0)

    tensor = torch.tensor(img_norm).float().to(DEVICE)

    with torch.no_grad():  
        pred = seg_model(tensor)

    pred = torch.sigmoid(pred).cpu().numpy()[0][0]

    mask = (pred > 0.5).astype(np.uint8) * 255  
    mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

    return mask

# -----------------------------
# FEATURE EXTRACTION
# -----------------------------
def extract_features(image, mask):
    sclera = cv2.bitwise_and(image, image, mask=mask)  
    pixels = sclera[mask > 0]

    if len(pixels) == 0:  
        return None

    mean_b = np.mean(pixels[:, 0])  
    mean_g = np.mean(pixels[:, 1])  
    mean_r = np.mean(pixels[:, 2])

    hsv = cv2.cvtColor(sclera, cv2.COLOR_BGR2HSV)  
    hsv_pixels = hsv[mask > 0]

    mean_h = np.mean(hsv_pixels[:, 0])  
    mean_s = np.mean(hsv_pixels[:, 1])  
    mean_v = np.mean(hsv_pixels[:, 2])

    lab = cv2.cvtColor(sclera, cv2.COLOR_BGR2LAB)  
    lab_pixels = lab[mask > 0]

    mean_l = np.mean(lab_pixels[:, 0])  
    mean_a = np.mean(lab_pixels[:, 1])  
    mean_b_lab = np.mean(lab_pixels[:, 2])

    yellow_index = mean_r - mean_b

    features = [  
        mean_r, mean_g, mean_b,  
        mean_h, mean_s, mean_v,  
        mean_l, mean_a, mean_b_lab,  
        yellow_index  
    ]

    return np.array(features).reshape(1, -1)

# -----------------------------
# SCLERA OVERLAY
# -----------------------------
def create_overlay(image, mask):
    overlay = image.copy()  
    overlay[mask > 0] = [0, 255, 0]  

    blended = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)

    return blended

# -----------------------------
# UI
# -----------------------------
st.title("Non-Invasive Jaundice Detection using Sclera Images")

st.write(
    "This prototype system detects jaundice from eye images by segmenting the sclera region "
    "using deep learning and estimating bilirubin levels using machine learning."
)

uploaded_file = st.file_uploader("Upload Eye Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Load via PIL, convert to NumPy array, then convert RGB to BGR for OpenCV
    pil_image = Image.open(uploaded_file)  
    image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    if st.button("Run Detection"):
        mask = segment_sclera(image)
        sclera = cv2.bitwise_and(image, image, mask=mask)
        overlay = create_overlay(image, mask)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Uploaded Image")
            # Convert back to RGB for Streamlit display
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_column_width=True)

        with col2:
            st.subheader("Sclera Mask")
            st.image(mask, use_column_width=True)

        with col3:
            st.subheader("Sclera Overlay")
            st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), use_column_width=True)

        features = extract_features(image, mask)

        if features is None:
            st.error("Sclera region could not be detected properly.")
            st.stop()

        features = scaler.transform(features)
        prob = classifier.predict_proba(features)[0][1]

        st.subheader("Prediction Results")
        st.write("Jaundice Probability:", round(prob, 3))
        st.progress(float(prob))

        THRESHOLD = 0.8

        if prob < THRESHOLD:
            st.success("Prediction: NORMAL")  
            st.write("Estimated Bilirubin: < 2 mg/dL")
        else:
            bilirubin = regressor.predict(features)[0]

            st.error("Prediction: JAUNDICE DETECTED")  
            st.write("Estimated Bilirubin:", round(bilirubin, 2), "mg/dL")

            st.subheader("Clinical Support Information")
            st.write("""
Possible medications commonly used in jaundice management:

• Ursodeoxycholic acid
• Silymarin
• Vitamin B complex

⚠ This system provides decision-support information only.
Please consult a qualified physician for diagnosis and treatment.
""")