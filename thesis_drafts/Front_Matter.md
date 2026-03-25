# TITLE PAGE

**AI-BASED NON-INVASIVE JAUNDICE DETECTION USING SCLERA IMAGE ANALYSIS**

A Thesis Submitted to the
School of Engineering
St Aloysius College (Deemed to Be University)

In Partial Fulfillment of the Requirements for the Degree of
**MASTER OF SCIENCE (MSC) IN SOFTWARE TECHNOLOGY**

by
**ASHITH JOSWA FERNANDES**
(Register No: 24251106)

Under the Guidance of
**DR. RUBAN S**
Dean and Associate Professor, School of Engineering

**ST ALOYSIUS COLLEGE (DEEMED TO BE UNIVERSITY)**
Mangalore, Karnataka, India
MARCH 2026

---

# CERTIFICATE

This is to certify that the thesis entitled **"AI-BASED NON-INVASIVE JAUNDICE DETECTION USING SCLERA IMAGE ANALYSIS"** submitted by **ASHITH JOSWA FERNANDES** is a record of original research work carried out by him under my supervision and guidance. The content of this thesis has not been submitted previously for the award of any degree, diploma, or title of any University.

**DR. RUBAN S**
(Project Guide)
Dean, School of Engineering
St Aloysius College

---

# DECLARATION

I, **ASHITH JOSWA FERNANDES**, hereby declare that the work presented in this thesis, entitled **"AI-BASED NON-INVASIVE JAUNDICE DETECTION USING SCLERA IMAGE ANALYSIS"**, is a result of my own research carried out under the guidance of Dr. Ruban S. This thesis does not contain any material previously published by another person, except where due reference is made in the text.

**ASHITH JOSWA FERNANDES**

---

# ACKNOWLEDGEMENT

I would like to express my sincere gratitude and respect to my guide, **Dr. Ruban S**, for his invaluable guidance, continuous encouragement, and technical insights throughout the course of this research. His mentorship has been instrumental in the completion of this project.

I am also thankful to the management and staff of **St Aloysius College (Deemed to be University)** and the **AIMIT campus** for providing for providing the facilities and resources required for my studies.

I express my deep appreciation to **Father Muller Medical College Hospital** for providing the clinical dataset and validation support that formed the foundation of this research.

Finally, I want to thank my family and friends for their constant support and motivation.

# SYNOPSIS

## Title of the Project
**AI-BASED NON-INVASIVE JAUNDICE DETECTION USING SCLERA IMAGE ANALYSIS**

## Statement of the Problem
Serum bilirubin is an important biochemical marker used in diagnosing and monitoring liver-related disorders such as jaundice. Conventional bilirubin estimation methods require invasive blood sampling, which can be uncomfortable, time-consuming, and unsuitable for frequent monitoring.

Clinical observations show that elevated bilirubin levels cause visible yellowing of the sclera (white part of the eye), a condition known as scleral icterus. With recent advances in image processing and machine learning, it is possible to analyze scleral color information from eye images to estimate bilirubin levels in a non-invasive manner.

The problem addressed in this project is the absence of a simple, explainable, and non-invasive computational approach for estimating serum bilirubin using eye images while ensuring ethical safety, interpretability, and feasibility for real-world screening applications.

## Objectives
- To develop a non-invasive approach for estimating serum bilirubin levels using eye images.
- To isolate the scleral region using image processing techniques.
- To extract color-based scleral features relevant to bilirubin estimation.
- To apply machine learning regression models for bilirubin prediction.
- To design a simple frontend interface to demonstrate system functionality.

## Methodology
The project follows a structured and modular methodology:
1. **Image Acquisition**: Eye images are collected from ethically approved and anonymized patient data.
2. **Preprocessing**: Images are resized and converted into suitable color spaces for analysis.
3. **Sclera Segmentation**: HSV-based color thresholding and deep learning-based segmentation (U-Net) are applied to isolate the scleral region.
4. **Feature Extraction**: Color features such as mean RGB values and Yellow Index are extracted from the segmented sclera.
5. **Machine Learning Model**: Regression models (Random Forest, XGBoost) are trained using scleral features and corresponding serum bilirubin values.
6. **Frontend Demonstration**: A Streamlit-based frontend allows users to upload eye images, visualize sclera extraction, and view estimated bilirubin values.

## Conclusion
This project demonstrates the feasibility of estimating serum bilirubin levels using scleral image analysis and machine learning in a non-invasive manner. The developed prototype integrates image processing, feature extraction, regression modeling, and frontend visualization into a unified system. While the current implementation serves as a validated feasibility study, it establishes a strong foundation for future clinical validation and system enhancement.

---

# ABSTRACT

Jaundice is a common medical condition resulting from hyperbilirubinemia, leading to yellowing of the skin and eyes. Traditional diagnosis requires invasive blood sampling, which is often inaccessible and slow in low-resource settings. This thesis proposes a non-invasive, AI-driven diagnostic system for jaundice detection specifically using eye sclera image analysis. 

The proposed system follows a robust modular pipeline: (1) Color normalization via the Gray World algorithm, (2) Deep learning-based sclera segmentation using a U-Net architecture with a ResNet34 encoder, and (3) A two-stage predictive modeling framework. The first stage uses a Random Forest classifier for binary jaundice detection, achieving an accuracy of ~90% and a sensitivity of 92%. The second stage employs an XGBoost regressor for serum bilirubin level estimation, achieving a Mean Absolute Error (MAE) of 1.2-1.8 mg/dL. 

A user-friendly Streamlit web interface was developed to demonstrate the system's real-time screening capabilities. This research demonstrates the feasibility of high-accuracy, non-invasive hyperbilirubinemia screening, offering a scalable solution for early diagnosis and remote health monitoring.

**Keywords**: Jaundice, Sclera Segmentation, U-Net, Random Forest, XGBoost, Non-Invasive Diagnostics, Computer Vision.
