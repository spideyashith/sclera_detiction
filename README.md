# 👁️ Non-Invasive Jaundice Detection using Eye Sclera Images and Machine Learning

> A research-grade AI screening tool for jaundice detection and bilirubin estimation using sclera (eye white) images — no blood test required.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Research%20Prototype-yellow)

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Medical Background](#medical-background)
- [Dataset Description](#dataset-description)
- [Methodology](#methodology)
- [Model Architecture](#model-architecture)
- [Results & Performance](#results--performance)
- [Confusion Matrix Explanation](#confusion-matrix-explanation)
- [Key Insights](#key-insights)
- [Challenges & Limitations](#challenges--limitations)
- [Future Improvements](#future-improvements)
- [Disclaimer](#disclaimer)

---

## 🔬 Project Overview

This project presents a **non-invasive, machine learning-based screening system** for detecting jaundice and estimating serum bilirubin levels from **sclera (eye white) images**. Jaundice manifests as yellowing of the sclera due to elevated bilirubin — a visually detectable biomarker that this system leverages through automated image analysis.

The system is designed as an accessible screening prototype, particularly relevant in settings where blood-based laboratory testing may be slow, costly, or unavailable. The core pipeline involves sclera segmentation, feature extraction, binary jaundice classification, and — if jaundice is detected — bilirubin level regression.

**Key Highlights:**
- Non-invasive: works from a photograph of the eye
- High recall (~97%) to minimize missed diagnoses
- Two-stage architecture: classification → regression
- Real patient data with ground-truth serum bilirubin values

---

## 🩺 Medical Background

**Jaundice** (icterus) is a clinical condition characterized by yellowing of the skin, mucous membranes, and sclera caused by elevated **bilirubin** in the bloodstream. Bilirubin is a yellow pigment produced during the normal breakdown of red blood cells.

| Bilirubin Level | Clinical Interpretation |
|---|---|
| < 1.2 mg/dL | Normal |
| 1.2 – 3.0 mg/dL | Mild elevation (subclinical) |
| > 3.0 mg/dL | Clinical jaundice (visible yellowing) |
| > 10 mg/dL | Severe jaundice |

Traditional jaundice diagnosis requires **serum bilirubin blood tests**, which involve venipuncture, lab equipment, and processing time. Non-invasive alternatives (e.g., transcutaneous bilirubinometers) exist but are expensive and limited in availability.

The **sclera** is an ideal non-invasive biomarker site because:
- It has minimal melanin pigmentation, reducing confounding skin-tone variation
- Bilirubin deposition causes a distinct yellowish shift detectable in the yellow/hue color channels
- It is easily photographable using standard cameras or smartphones

---

## 📁 Dataset Description

| Property | Detail |
|---|---|
| Total Patients Enrolled | 36 |
| Raw Images Collected | 126+ real sclera images |
| Final Dataset Size (post-augmentation) | 211 samples |
| Ground Truth | Serum bilirubin value from blood test (mg/dL) |
| Jaundice Positive (~Bilirubin > 3 mg/dL) | ~70% of samples |
| Normal (~Bilirubin ≤ 3 mg/dL) | ~30% of samples |

**Data Augmentation Techniques Applied:**
- Horizontal and vertical flipping
- Brightness and contrast perturbation
- Rotation and zoom transformations

**Class Imbalance Handling:**
- SMOTE (Synthetic Minority Over-sampling Technique) applied on the training split to balance class distribution before model training

> ⚠️ Dataset is not publicly released to protect patient privacy. All data was collected under appropriate institutional guidelines.

---

## ⚙️ Methodology

### Stage 1 — Sclera Extraction

The sclera region is isolated from raw eye images using a classical computer vision pipeline:

```
Raw Eye Image
     │
     ▼
RGB → HSV Color Conversion
     │
     ▼
HSV Thresholding (isolate white/off-white sclera region)
     │
     ▼
Morphological Operations (Erosion → Dilation)
     │  (removes noise, fills gaps)
     ▼
Largest Connected Component Selection
     │  (keeps dominant sclera blob)
     ▼
Extracted Sclera Mask
```

### Stage 2 — Feature Engineering

From the extracted sclera region, the following features are computed:

| Feature | Description | Domain |
|---|---|---|
| Mean R, G, B | Average red, green, blue channel values | Color |
| Hue (H) | Dominant color tone in HSV space | Color |
| Saturation (S) | Color purity/intensity | Color |
| Value (V) | Brightness of the region | Color |
| Entropy | Shannon entropy measuring texture complexity | Texture |
| **Yellow Index (LAB)** | **b* channel from CIE-LAB space; primary jaundice indicator** | **Color** |

The **Yellow Index** quantifies the degree of yellow pigmentation in the sclera using the `b*` axis of the perceptually uniform CIE-LAB color space, where higher values indicate stronger yellow bias.

---

## 🏗️ Model Architecture

This system employs a **Two-Stage ML Pipeline** designed to mimic clinical decision-making:

```
Input Sclera Image
        │
        ▼
  Feature Extraction (8 features)
        │
        ▼
┌───────────────────────────────┐
│  STAGE 1: Binary Classifier   │
│  Model: XGBoost               │
│  Output: Jaundice / Normal    │
└───────────────────────────────┘
        │
   ┌────┴────┐
   │         │
Normal     Jaundice Detected
   │         │
   │         ▼
   │  ┌──────────────────────────────┐
   │  │  STAGE 2: Bilirubin Regressor│
   │  │  Output: Estimated mg/dL     │
   │  └──────────────────────────────┘
   │         │
   └────┬────┘
        ▼
  Final Report
```

**Why Two Stages?**
- Regression is only triggered for positive cases, reducing noise from normal samples influencing the bilirubin estimate
- Mirrors clinical workflow: first screen, then quantify

**Why XGBoost over Random Forest?**
- Superior performance on small, imbalanced tabular datasets
- Built-in regularization reduces overfitting
- Better calibrated probability outputs for threshold tuning
- More interpretable feature importance scores

---

## 📊 Results & Performance

### Stage 1 — Classification (XGBoost)

| Metric | Value |
|---|---|
| Accuracy | ~90% |
| Recall (Sensitivity) | ~97% |
| ROC-AUC | ~0.71 |
| False Negative Rate | ~3% |

> **Recall is the primary metric of concern in this medical context.** A false negative (missed jaundice) carries significantly higher risk than a false positive (unnecessary follow-up). The model is tuned to prioritize sensitivity.

### Stage 2 — Bilirubin Regression

| Metric | Value | Clinical Context |
|---|---|---|
| MAE | 2.69 mg/dL | Mean absolute error in bilirubin estimate |
| RMSE | 3.72 mg/dL | Root mean squared error (penalizes outliers) |

The regression error is clinically acceptable for **screening-level triage** but insufficient for precise clinical management. Estimated bilirubin values should be used to flag patients for confirmatory blood tests rather than for treatment decisions.

---

## 🔢 Confusion Matrix Explanation

For Stage 1 Classification (Test Set):

```
                     Predicted
                  Normal    Jaundice
Actual  Normal  [  TN=18  |  FP=42  ]
        Jaundice[  FN=2   |  TP=58  ]
```

| Cell | Count | Meaning |
|---|---|---|
| **True Positive (TP)** | 58 | Jaundice correctly identified ✅ |
| **True Negative (TN)** | 18 | Normal correctly identified ✅ |
| **False Negative (FN)** | 2 | Jaundice missed — **most critical error** ⚠️ |
| **False Positive (FP)** | 42 | Normal flagged as jaundice — leads to unnecessary follow-up |

**Interpretation:**

The high false positive count (FP=42) reflects the deliberate trade-off made by lowering the classification threshold to minimize false negatives. In a screening context, it is far safer to flag a healthy patient for follow-up (FP) than to miss an actual jaundice case (FN). The very low FN count (2) demonstrates the model's strong sensitivity, which is the primary clinical requirement for a screening tool.

---

## 💡 Key Insights

**1. Yellow Index is the Dominant Predictive Feature**
The LAB-based Yellow Index accounts for approximately **36% of the total feature importance** in the XGBoost model — confirming that the `b*` (blue-yellow) channel of the CIE-LAB color space is the most diagnostically meaningful signal for jaundice detection.

**2. High Recall by Design**
The classification threshold was explicitly tuned to achieve ~97% recall, reflecting the medical principle that *sensitivity must be prioritized in screening applications*. The cost of missing a jaundice case (FN) outweighs the cost of a false alarm (FP).

**3. Two-Stage Architecture Improves Regression Quality**
By restricting bilirubin estimation to Stage 1 positive cases only, the regression model is trained on a more homogeneous subset, reducing noise and improving estimate validity.

**4. HSV and LAB Color Spaces Complement Each Other**
HSV features capture gross color and saturation characteristics, while the LAB Yellow Index captures the perceptually significant yellow shift. Together they provide a richer representation than either space alone.

**5. Sclera is a Reliable Biomarker Site**
The minimal melanin in the sclera makes it more consistent across different skin tones than skin-based methods, improving generalizability.

---

## ⚠️ Challenges & Limitations

| Challenge | Impact | Mitigation Applied |
|---|---|---|
| Small dataset (36 patients) | High risk of overfitting, limited generalizability | Data augmentation + SMOTE |
| Class imbalance (~70/30) | Biased model toward majority class | SMOTE on training data, threshold tuning |
| Limited normal samples | Weak decision boundary for normal class | Augmentation of normal class |
| Lighting and camera variation | Feature inconsistency across images | Normalized color features, HSV space |
| Imperfect sclera segmentation | Noise in extracted features | Morphological post-processing, largest-component selection |
| No demographic diversity control | Potential bias across age/ethnicity | Future work — diverse dataset collection |

---

## 🚀 Future Improvements

**Data & Validation**
- Expand dataset substantially — target 500+ patients with balanced normal/jaundice distribution
- Include demographic diversity (age, ethnicity, gender) for model fairness evaluation
- Conduct formal clinical validation in hospital settings

**Segmentation**
- Replace classical HSV thresholding with a deep learning segmentation model (e.g., **U-Net**) for more robust and accurate sclera extraction
- Explore iris/pupil exclusion refinement for cleaner sclera masks

**Modeling**
- Investigate deep learning end-to-end approaches (e.g., CNNs trained directly on sclera images)
- Ensemble classical features with learned CNN features
- Explore calibrated probability estimation for clinical risk scoring

**Deployment**
- Develop a **mobile application** for point-of-care screening in resource-limited settings
- Implement standardized image capture protocol to reduce lighting variability
- Integrate with electronic health record (EHR) systems

---

## ⚕️ Disclaimer

> **This project is a research prototype and is NOT intended for clinical use.**
>
> The system has been developed and validated on a small dataset of 36 patients and has not undergone formal clinical trials or regulatory review. It is designed solely as a **screening aid and proof-of-concept**, not as a replacement for laboratory serum bilirubin testing.
>
> **Do not use this tool to make clinical decisions.** Any individual showing signs of jaundice should be evaluated by a qualified healthcare professional and undergo appropriate laboratory investigations.
>
> The authors make no warranty, expressed or implied, regarding the accuracy, completeness, or fitness for any medical purpose of this software.

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

Special thanks to the patients who consented to participate, the clinical staff who assisted with sample collection and bilirubin assay, and the open-source scientific Python community whose tools made this work possible.

---

*For questions or collaboration inquiries, please open an issue or contact the project maintainer.*
