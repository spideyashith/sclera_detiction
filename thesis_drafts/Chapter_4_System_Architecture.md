# CHAPTER 4: SYSTEM ARCHITECTURE

## 4.1 Overview of the Diagnostic Pipeline
The proposed system follows a modular, two-stage pipeline designed for robustness, accuracy, and medical interpretability. The architecture ensures that each stage—from initial image acquisition to the final bilirubin level estimation—is optimized for its specific task while maintaining a seamless and efficient data flow. This modular design also facilitates independent testing and validation of each component, which is a critical requirement for medical-grade software.

## 4.2 High-Level Architecture
The system is logically divided into four primary modules, each responsible for a distinct phase of the diagnostic process:

1.  **Preprocessing & Normalization Module**: Focuses on stabilizing the input data by removing ambient lighting bias and sensor-specific color casts. This ensures that the downstream models receive consistent color information.
2.  **Segmentation Module**: Utilizes a deep learning U-Net model with a ResNet34 backbone to isolate the scleral region from raw eye images. This module is responsible for identifying the "Region of Interest" (ROI).
3.  **Feature Extraction & Engineering Module**: Processes the segmented ROI to compute a high-dimensional feature vector across multiple color spaces (RGB, HSV, LAB). This module also computes custom-engineered features like the Yellow Index.
4.  **Two-Stage Predictive Modeling Module**: Implements a hierarchical decision framework consisting of a binary classifier (Jaundice vs. Normal) and a specialized regressor for bilirubin concentration estimation.

## 4.3 Detailed Pipeline Components

### 4.3.1 Image Acquisition and Color Normalization
The input to the system is an RGB ocular image captured via a standard smartphone camera or professional medical imaging device. Because these images can be taken under a wide variety of lighting conditions (e.g., warm incandescent light, cool fluorescent light, or natural sunlight), the first step in the pipeline is **Color Normalization**.

We implement the **Gray World Algorithm**, which assumes that the average reflectance of a scene under achromatic lighting is gray. By calculating the mean intensities of the Red, Green, and Blue channels and scaling them toward a global gray value, we can effectively "neutralize" the lighting environment. This step is critical because bilirubin detection relies on the analysis of subtle yellow shifts in the sclera, which could otherwise be masked or exaggerated by the color temperature of the ambient light source.

### 4.3.2 Deep Learning-Based Sclera Segmentation
Earlier jaundice detection systems often relied on manual thresholding or simple edge detection, which were prone to failure in the presence of eyelashes, shadows, or varying eye shapes. To overcome these limitations, our architecture employs a **U-Net with a ResNet34 encoder**.

This module performs pixel-level classification to generate a binary mask where the sclera region is highlighted in white and all other tissues (skin, iris, pupil, and background) are masked in black. The use of a pretrained ResNet34 encoder allows the model to leverage complex spatial hierarchies, making it robust against occlusions from eyelids and eyelashes. The output of this module is a precise, isolated ROI that is used for subsequent color analysis.

### 4.3.3 Masking and Feature Vector Computation
Once the binary mask is generated, the system performs a bitwise-AND operation to extract only the scleral pixels from the normalized image. The system then computes a comprehensive 10-dimensional feature vector from this extracted region:

-   **RGB Space**: Mean intensities of Red, Green, and Blue channels. These provide the fundamental color composition of the sclera.
-   **HSV (Hue, Saturation, Value)**: These features are less sensitive to lighting intensity changes. The Hue channel, in particular, is a strong indicator of the "yellow" spectral component.
-   **CIELAB Space**: Contains L* (Lightness), a* (Green-Red axis), and b* (Blue-Yellow axis). The b* component is the most significant biomarker for bilirubin, as it directly quantifies the intensity of yellowing.
-   **Yellow Index (YI)**: A custom-engineered feature calculated as \( (R - B) / (R + B + G) \). This ratio highlights the dominance of the red/yellow spectrum over blue, which is a hallmark of hyperbilirubinemia.

### 4.3.4 Two-Stage Predictive Modeling Framework
The final decision-making module implements a hierarchical logic to ensure clinical relevance:

-   **Stage 1: Random Forest Classifier**: This model serves as the primary screening gate. It analyzes the 10-dimensional feature vector to determine if the patient belongs to the "Healthy" or "Jaundice" category. We use an ensemble of 500 decision trees to ensure high sensitivity, as it is better to have a false positive than to miss a case of jaundice.
-   **Stage 2: XGBoost Regressor**: If Stage 1 identifies the presence of jaundice, the system activates the second stage. This regressor is specifically trained on skewed pathological data to estimate the actual serum bilirubin concentration in mg/dL. By focusing only on jaundiced cases, the model can more accurately distinguish between mild and severe clinical states.

## 4.4 Data Flow and Sequence
The data flow within the system follows a strictly sequential and deterministic path:

1.  **Input**: Raw RGB Image.
2.  **Step 1**: Gray World Color Normalization.
3.  **Step 2**: U-Net Semantic Segmentation.
4.  **Step 3**: Binary Mask Generation & Application.
5.  **Step 4**: Statistical Color Feature Extraction (RGB, HSV, LAB).
6.  **Step 5**: Feature Scaling (StandardScaler).
7.  **Step 6**: Random Forest Binary Classification.
8.  **Step 7**: (If Jaundice) XGBoost Bilirubin Level Regression.
9.  **Output**: Diagnostic result, confidence score, and estimated bilirubin level.

## 4.5 Modularity, Scalability, and Extensibility
The architecture is designed to be highly extensible. Because each module is independent, they can be updated or replaced as technology advances:
-   The **Segmentation** module can be upgraded to more advanced architectures like V-Net or Swin-Transformer without affecting the classification logic.
-   The **Modeling** module can be retrained on larger or more diverse multi-ethnic datasets to improve generalizability.
-   The **Interface** can be ported from a web-based Streamlit app to a native mobile application or integrated into hospital management systems via API endpoints.

## 4.6 Architecture Diagram
[INSERT_ARCHITECTURE_DIAGRAM_HERE]
*Figure 4.1: High-Level System Architecture showing modular components from preprocessing to predictive modeling.*

## 4.7 Pipeline Diagram
[INSERT_PIPELINE_DIAGRAM_HERE]
*Figure 4.2: Detailed Diagnostic Pipeline representing data transformations across sequential processing stages.*

This design philosophy ensures that the system remains at the cutting edge of both computer vision and medical AI research.
