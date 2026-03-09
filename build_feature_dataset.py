import os
import cv2
import torch
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMG_SIZE = 256

IMAGE_ROOT = "images"
MODEL_PATH = "sclera_segmentation_model.pth"
LABEL_FILE = "labels.csv"

OUTPUT_FILE = "features_dataset.csv"

# load model
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=3,
    classes=1
)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()


def find_image(filename):
    for root, dirs, files in os.walk(IMAGE_ROOT):
        if filename in files:
            return os.path.join(root, filename)
    return None


def segment_sclera(image):

    img = cv2.resize(image,(IMG_SIZE,IMG_SIZE))
    img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    img_norm = img_rgb/255.0
    img_norm = np.transpose(img_norm,(2,0,1))
    img_norm = np.expand_dims(img_norm,0)

    tensor = torch.tensor(img_norm).float().to(DEVICE)

    with torch.no_grad():
        pred = model(tensor)

    pred = torch.sigmoid(pred).cpu().numpy()[0][0]

    mask = (pred>0.5).astype(np.uint8)*255
    mask = cv2.resize(mask,(image.shape[1],image.shape[0]))

    return mask


def extract_features(image, mask):

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

    return [
        mean_r,mean_g,mean_b,
        mean_h,mean_s,mean_v,
        mean_l,mean_a,mean_b_lab,
        yellow_index
    ]


labels = pd.read_csv(LABEL_FILE)

rows = []

for i,row in labels.iterrows():

    image_name = row["image"]
    bilirubin = row["bilirubin"]

    path = find_image(image_name)

    if path is None:
        print("Image not found:",image_name)
        continue

    image = cv2.imread(path)

    mask = segment_sclera(image)

    feats = extract_features(image,mask)

    if feats is None:
        continue

    rows.append(feats+[bilirubin,image_name])


columns = [
"mean_r","mean_g","mean_b",
"mean_h","mean_s","mean_v",
"mean_l","mean_a","mean_b_lab",
"yellow_index",
"bilirubin",
"image"
]

df = pd.DataFrame(rows,columns=columns)

df.to_csv(OUTPUT_FILE,index=False)

print("Feature dataset created:",OUTPUT_FILE)
print("Total samples:",len(df))