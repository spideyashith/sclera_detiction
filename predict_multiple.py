import cv2
import torch
import numpy as np
import joblib
import pandas as pd
import os
import segmentation_models_pytorch as smp

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMG_SIZE = 256

MODEL_PATH = "sclera_segmentation_model.pth"

classifier = joblib.load("jaundice_classifier.pkl")
regressor = joblib.load("bilirubin_regressor.pkl")

seg_model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=3,
    classes=1
)

seg_model.load_state_dict(torch.load(MODEL_PATH,map_location=DEVICE))
seg_model.to(DEVICE)
seg_model.eval()

def gray_world_normalization(img):

    img = img.astype(np.float32)

    avg_b = np.mean(img[:,:,0])
    avg_g = np.mean(img[:,:,1])
    avg_r = np.mean(img[:,:,2])

    avg_gray = (avg_b + avg_g + avg_r) / 3

    img[:,:,0] *= avg_gray / avg_b
    img[:,:,1] *= avg_gray / avg_g
    img[:,:,2] *= avg_gray / avg_r

    img = np.clip(img,0,255)

    return img.astype(np.uint8)

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

    mask = (pred>0.5).astype(np.uint8)*255
    mask = cv2.resize(mask,(image.shape[1],image.shape[0]))

    return mask

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
    print(features_df)



TEST_FOLDER = "test_images"

feature_names = [
"mean_r","mean_g","mean_b",
"mean_h","mean_s","mean_v",
"mean_l","mean_a","mean_b_lab",
"yellow_index"
]

THRESHOLD = 0.70

for file in os.listdir(TEST_FOLDER):

    path = os.path.join(TEST_FOLDER,file)

    image = cv2.imread(path)

    if image is None:
        continue

    image = gray_world_normalization(image)

    mask = segment_sclera(image)

    features = extract_features(image,mask)

    if features is None:
        print(file,"→ sclera not detected")
        continue

    features_df = pd.DataFrame(features,columns=feature_names)

    prob = classifier.predict_proba(features_df)[0][1]

    if prob < THRESHOLD:

        result = "NORMAL"

    else:

        result = "JAUNDICE"

    print(file,"→",result,"| probability:",round(prob,3))