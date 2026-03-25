# CHAPTER 3

# MATERIALS, METHODS AND METHODOLOGY

## 3.1 INTRODUCTION

The core of this project lies in the development of a robust pipeline for the automated, non-invasive detection of jaundice through the digital analysis of sclera images. This process involves a transition from raw anatomical data to refined diagnostic insights. The methodology is structured to be reproducible, academically rigorous, and scalable for potential mobile deployment. This chapter elaborates on each stage of the system architecture, including data acquisition, preprocessing, deep learning-based segmentation, multi-space feature extraction, and the training of classification and regression models.

## 3.2 DATA SOURCES & FORMAT

The dataset utilized in this study consists of high-resolution digital eye images collected from both healthy individuals and patients diagnosed with various stages of jaundice (hyperbilirubinemia). To ensure a balanced and robust model, data was sourced from multiple clinical environments and publicly available medical image repositories.

-   **Image Format**: All raw images were captured in JPEG or PNG format at resolutions ranging from 2 megapixels to 12 megapixels.
-   **Lighting Conditions**: The dataset includes images captured under different lighting scenarios (natural daylight, fluorescent indoor lighting, and camera flash) to allow the model to learn invariance to color temperature shifts.
-   **Diversity**: Images represent a wide range of ages, genders, and ethnicities to ensure that variations in skin tone and eyelid morphology do not bias the sclera segmentation and color analysis.
-   **Ground Truth**: For each patient image, the corresponding Serum Bilirubin (SBR) level was recorded as the gold-standard reference. Labels were categorized into 'Healthy' (SBR < 1.2 mg/dL) and 'Jaundiced' (SBR >= 1.2 mg/dL) for classification tasks.

## 3.3 DATA PREPROCESSING & AUGMENTATION

Before feeding the images into the neural network, several preprocessing steps were implemented to ensure uniformity and improve model generalization:

1.  **Resizing**: All images were resized to a standard resolution of 256x256 pixels to maintain a balance between computational efficiency and the retention of critical color and texture details.
2.  **Normalization**: Pixel intensities were normalized to a range of [0, 1] to stabilize the training process of the U-Net segmentation model.
3.  **Augmentation**: To prevent overfitting and enhance the model's robustness, the training set was augmented using several techniques:
    -   **Random Horizontal Flipping**: To account for left and right eye variations.
    -   **Rotation and Zooming**: To simulate different camera angles and distances.
    -   **Brightness and Contrast Adjustments**: To mimic various environmental lighting conditions.

## 3.4 FEATURE EXTRACTION (DETAILED)

Feature extraction is the process of converting the segmented sclera region into a set of numerical descriptors that represent the jaundice-indicative color properties. While humans perceive jaundice as "yellowing," digital sensors capture this as a complex interaction across different color channels.

### 3.4.1 Color Space Selection
To capture the most relevant information, we utilized three distinct color spaces:
-   **RGB (Red, Green, Blue)**: The default additive color model used by digital cameras.
-   **CIELAB (L*a*b*)**: A perceptually uniform color space. The 'b*' channel represents the yellow-blue axis, making it the most critical biomarker for bilirubin detection.
-   **HSV (Hue, Saturation, Value)**: Useful for separating color information (Hue) from intensity (Value), helping to mitigate the effects of shadow and highlights.

### 3.4.2 Implementation: Feature Extraction Code

The following Python implementation demonstrates the extraction of mean channel values and custom ratios from the segmented sclera region.

```python
import cv2
import numpy as np
import pandas as pd
import os

def extract_sclera_features(image_path, mask_path=None):
    # Load the eye image
    img = cv2.imread(image_path)
    if img is None: return None
    
    img = cv2.resize(img, (256, 256))
    
    # If a mask isn't provided, we assume a simple threshold or external mask
    # For extraction purposes, we consider non-black pixels as the sclera region
    mask = np.sum(img, axis=2) > 0
    pixels = img[mask]
    
    if len(pixels) < 50: return None
    
    # RGB Features
    mean_r = np.mean(pixels[:, 2])
    mean_g = np.mean(pixels[:, 1])
    mean_b = np.mean(pixels[:, 0])
    
    # CIELAB Features
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab_pixels = lab[mask]
    mean_l = np.mean(lab_pixels[:, 0])
    mean_a = np.mean(lab_pixels[:, 1])
    mean_b_lab = np.mean(lab_pixels[:, 2]) # Yellow-Blue axis
    
    # Ratios (Biomarkers)
    rg_ratio = mean_r / (mean_g + 1e-6)
    rb_ratio = mean_r / (mean_b + 1e-6)
    yellow_index = mean_b_lab / (mean_l + 1e-6)
    
    return {
        "mean_r": mean_r, "mean_g": mean_g, "mean_b": mean_b,
        "mean_l": mean_l, "mean_a": mean_a, "mean_b_lab": mean_b_lab,
        "yellow_index": yellow_index, "rg_ratio": rg_ratio, "rb_ratio": rb_ratio
    }
```

## 3.5 NEURAL NETWORK ARCHITECTURE & TRAINING

The system incorporates two primary neural architectures: a U-Net for segmentation and a multi-layer perceptron (MLP) for regression/classification.

### 3.5.1 Sclera Segmentation (U-Net)
U-Net is a convolutional neural network architecture developed for biomedical image segmentation. It consists of a contracting path (encoder) to capture context and a symmetric expanding path (decoder) that enables precise localization. The use of skip connections allows the model to preserve fine-grained spatial information which is crucial for identifying the boundaries of the sclera against the iris and eyelids.

### 3.5.2 Implementation: Training Pipeline Code (Segmenter)

The training of the segmentation model is performed using a combination of Binary Cross-Entropy and Dice Loss to handle imbalanced pixel distributions.

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

class ScleraSegmentationModel(nn.Module):
    def __init__(self):
        super(ScleraSegmentationModel, self).__init__()
        # Simple Encoder-Decoder for demonstration (U-Net style)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 1, kernel_size=2, stride=2),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Training Loop snippet
def train_segmenter(model, loader, criterion, optimizer, epochs=50):
    model.train()
    for epoch in range(epochs):
        for images, masks in loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
```

## 3.6 MODEL EVALUATION & VERSIONING

Evaluation is performed on a held-out test set to ensure the system's ability to generalize to unseen patients. We utilize multiple metrics to provide a 360-degree view of model performance.

### 3.6.1 Metrics for Classification
-   **Accuracy**: Overall percentage of correct predictions.
-   **Sensitivity (Recall)**: Ability to correctly identify all jaundiced cases (critical for medical screening).
-   **Specificity**: Ability to correctly identify healthy cases.
-   **F1-Score**: Harmonic mean of Precision and Recall.

### 3.6.2 Metrics for Bilirubin Regression
-   **Mean Absolute Error (MAE)**: The average difference between predicted and actual bilirubin levels.
-   **R-Squared (R²)**: The proportion of variance in bilirubin levels that is predictable from the image features.

## 3.7 PROCESS DESCRIPTION (INFERENCE PIPELINE)

The inference pipeline is designed to be a seamless "one-click" diagnostic process. It integrates all the aforementioned components into a single executable flow.

### 3.7.1 Implementation: Real-Time Prediction Code

```python
def predict_jaundice(image_path, seg_model, class_model, reg_model):
    # 1. Segment Sclera
    img = preprocess(image_path)
    mask = seg_model(img)
    
    # 2. Extract Features
    features = extract_sclera_features(image_path, mask)
    
    # 3. Classify and Estimate
    is_jaundiced = class_model.predict(features)
    bilirubin_level = reg_model.predict(features)
    
    return {
        "Status": "Jaundiced" if is_jaundiced else "Healthy",
        "Estimated Bilirubin": f"{bilirubin_level:.2f} mg/dL"
    }
```

## 3.8 WORKFLOW DIAGRAM

*[Insert Flowchart Here]*
The workflow begins with image acquisition, followed by preprocessing and U-Net-based segmentation. The segmented sclera is then passed through a feature extractor, which feeds into the classification and regression models to generate the final diagnostic report.

## 3.9 HARDWARE & SOFTWARE SPECIFICATIONS

### Hardware Requirements
-   **Processor**: Intel i7 or higher (AMD Ryzen 7 equivalent).
-   **Memory**: 16GB RAM minimum.
-   **GPU**: NVIDIA RTX 3060 or better with 6GB+ VRAM (for training).
-   **Storage**: 500GB SSD for dataset and model checkpoints.

### Software Requirements
-   **Operating System**: Windows 10/11 or Ubuntu 20.04.
-   **Languages**: Python 3.9+.
-   **Libraries**: PyTorch, OpenCV, Scikit-learn, Pandas, NumPy, Matplotlib.
-   **Tools**: Jupyter Notebooks, Git, and NVIDIA CUDA/cuDNN drivers.

---
