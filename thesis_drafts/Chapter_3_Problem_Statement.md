# CHAPTER 3: PROBLEM STATEMENT AND OBJECTIVES

## 3.1 Problem Statement
The diagnosis of jaundice currently relies on invasive blood tests to measure serum bilirubin levels. In many parts of the world, especially in rural and developing regions, the lack of immediate access to laboratory facilities often leads to delayed diagnosis. Furthermore, the invasive nature of blood draws creates discomfort and may discourage frequent monitoring in chronic liver disease patients.

Existing non-invasive methods, such as visual inspection and TcB, are either subjective or prohibitively expensive for large-scale preliminary screening. While AI-based image analysis shows promise, early implementations suffered from:
-   **Inconsistent Sclera Extraction**: Classical methods failed to differentiate between the sclera and surrounding skin or eyelids under varying illumination.
-   **Lighting Artifacts**: Ambient light color temperature directly affects the perceived color of the sclera, leading to false positives or inaccurate bilirubin estimations.
-   **Dataset Imbalance**: Limited availability of clinically validated patient images hinders the development of balanced and generalized models.

There is a critical need for a non-invasive, AI-driven screening system that can accurately isolate the sclera, normalize color variations, and provide reliable jaundice detection and bilirubin estimation.

## 3.2 Research Objectives
The primary goal of this research is to develop an automated diagnostic pipeline for jaundice detection. Specifically, the project aims to:
1.  **Develop a Precise Segmentation Framework**: Utilize a Deep Learning U-Net architecture to consistently extract the sclera region across diverse eye shapes and lighting conditions.
2.  **Implement Color Normalization**: Apply the Gray World algorithm to stabilize color features against varying ambient lighting temperatures.
3.  **Perform High-Dimensional Feature Extraction**: Analyze the sclera in multiple color spaces (RGB, HSV, LAB) to identify the most potent biomarkers for hyperbilirubinemia.
4.  **Construct a Two-Stage Diagnostic Model**:
    -   **Stage 1**: A binary classification model to identify the presence of jaundice.
    -   **Stage 2**: A regression model to provide a quantitative estimate of serum bilirubin levels for positive cases.
5.  **Evaluate and Validate Performance**: Assess the system using standard metrics including Sensitivity, Specificity, Accuracy, and Mean Absolute Error (MAE) against clinical ground truth data.
6.  **Create a Practical Interface**: Build a Streamlit-based web application to demonstrate the feasibility of the system for real-world screening scenarios.

## 3.3 Scope and Limitations
This study focuses on adults and utilizes a dataset collected from patients at the St Aloysius AI Research lab. The system is designed as a **screening tool** and is not intended to replace definitive clinical laboratory diagnosis. The current scope is limited to eye images captured in controlled or semi-controlled indoor environments.
