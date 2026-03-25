# APPENDIX A: SOURCE CODE

This appendix contains the complete source code utilized in the "AI-Based Non-Invasive Jaundice Detection" project. The code is organized by functional modules, including data preprocessing, segmentation modeling, feature extraction, and the diagnostic pipeline.

## A.1 Sclera Segmentation Training (U-Net)

This script implements the training pipeline for the U-Net model using the `segmentation_models_pytorch` library.

```python
import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import albumentations as A

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_DIR = "segmentation_dataset/images"
MASK_DIR = "segmentation_dataset/masks"
IMG_SIZE = 256
BATCH_SIZE = 4
EPOCHS = 20

class ScleraDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = os.listdir(image_dir)
        self.transform = A.Compose([A.Resize(IMG_SIZE, IMG_SIZE)])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name.replace(".jpg", ".png").replace(".jpeg",".png"))
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, 0)
        transformed = self.transform(image=image, mask=mask)
        image = transformed["image"] / 255.0
        mask = transformed["mask"] / 255.0
        image = np.transpose(image, (2, 0, 1))
        return torch.tensor(image).float(), torch.tensor(mask).unsqueeze(0).float()

dataset = ScleraDataset(IMAGE_DIR, MASK_DIR)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=1).to(DEVICE)
loss_fn = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for images, masks in loader:
        images, masks = images.to(DEVICE), masks.to(DEVICE)
        preds = model(images)
        loss = loss_fn(preds, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS} Loss:", total_loss)

torch.save(model.state_dict(), "sclera_segmentation_model.pth")
```

## A.2 Jaundice Classification Training (Random Forest)

This script trains a Random Forest classifier on color features extracted from the sclera to distinguish between jaundiced and healthy samples.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib

df = pd.read_csv("features_dataset.csv")
df["label"] = (df["bilirubin"] > 2).astype(int)
features = ["mean_r","mean_g","mean_b", "mean_h","mean_s","mean_v", "mean_l","mean_a","mean_b_lab", "yellow_index"]
X, y = df[features], df["label"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, "feature_scaler.pkl")

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
model = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")
model.fit(X_train, y_train)

pred = model.predict(X_test)
print(classification_report(y_test, pred))
joblib.dump(model, "jaundice_classifier.pkl")
```

## A.3 Real-Time Diagnostic Pipeline

This comprehensive script integrates the segmentation model and the classification/regression models into a single end-to-end inference flow.

```python
import cv2
import torch
import numpy as np
import joblib
import pandas as pd
import segmentation_models_pytorch as smp

scaler = joblib.load("feature_scaler.pkl")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 256
MODEL_PATH = "sclera_segmentation_model.pth"
classifier = joblib.load("jaundice_classifier.pkl")
regressor = joblib.load("bilirubin_regressor.pkl")

seg_model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1)
seg_model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
seg_model.to(DEVICE).eval()

def gray_world_normalization(img):
    img = img.astype(np.float32)
    avg_gray = (np.mean(img[:,:,0]) + np.mean(img[:,:,1]) + np.mean(img[:,:,2])) / 3
    img[:,:,0] *= (avg_gray / np.mean(img[:,:,0]))
    img[:,:,1] *= (avg_gray / np.mean(img[:,:,1]))
    img[:,:,2] *= (avg_gray / np.mean(img[:,:,2]))
    return np.clip(img, 0, 255).astype(np.uint8)

def segment_sclera(image):
    img = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    img_norm = np.transpose(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)/255.0, (2, 0, 1))
    tensor = torch.tensor(np.expand_dims(img_norm, 0)).float().to(DEVICE)
    with torch.no_grad():
        pred = torch.sigmoid(seg_model(tensor)).cpu().numpy()[0][0]
    mask = (pred > 0.5).astype(np.uint8) * 255
    return cv2.resize(mask, (image.shape[1], image.shape[0]))

def extract_features(image, mask):
    sclera = cv2.bitwise_and(image, image, mask=mask)
    pixels = sclera[mask > 0]
    if len(pixels) == 0: return None
    hsv_pixels = cv2.cvtColor(sclera, cv2.COLOR_BGR2HSV)[mask > 0]
    lab_pixels = cv2.cvtColor(sclera, cv2.COLOR_BGR2LAB)[mask > 0]
    return np.array([np.mean(pixels[:,2]), np.mean(pixels[:,1]), np.mean(pixels[:,0]),
                     np.mean(hsv_pixels[:,0]), np.mean(hsv_pixels[:,1]), np.mean(hsv_pixels[:,2]),
                     np.mean(lab_pixels[:,0]), np.mean(lab_pixels[:,1]), np.mean(lab_pixels[:,2]),
                     np.mean(pixels[:,2]) - np.mean(pixels[:,0])]).reshape(1, -1)

image = cv2.imread("jud_eye.jpg")
image = gray_world_normalization(image)
mask = segment_sclera(image)
features = extract_features(image, mask)
if features is not None:
    features_scaled = scaler.transform(pd.DataFrame(features, columns=["mean_r","mean_g","mean_b","mean_h","mean_s","mean_v","mean_l","mean_a","mean_b_lab","yellow_index"]))
    prob = classifier.predict_proba(features_scaled)[0][1]
    print("Jaundice Probability:", prob)
    if prob >= 0.75:
        print("Bilirubin:", regressor.predict(features_scaled)[0])
```
