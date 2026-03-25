# CHAPTER 10: SYSTEM INTERFACE

## 10.1 Interface Design Objectives
The primary objective of the system interface was to provide a simple, intuitive, and medically informative dashboard for clinicians and healthcare workers. The interface serves as a bridge between the complex underlying AI models and the end-user.

## 10.2 Streamlit Web Application
The interface is developed using **Streamlit**, a Python-based framework specifically designed for data science and machine learning applications.

### 10.2.1 Core Features of the Interface
-   **Image Uploader**: A drag-and-drop file uploader that accepts eye images in common formats (.jpg, .png, .jpeg).
-   **Automated Inference**: Upon uploading an image, the system automatically triggers the segmentation, feature extraction, classification, and regression pipeline.
-   **Visualization Panel**: The application displays the processed image with a semi-transparent green overlay highlighting the segmented sclera. This provides the user with visual evidence that the model is analyzing the correct anatomical region.
-   **Diagnostic Metrics**:
    -   **Jaundice Probability**: A percentage score representing the model's confidence.
    -   **Prediction Label**: Categorizes the case as "Normal" or "Jaundice Detected" based on the probability threshold.
    -   **Estimated Bilirubin**: Provides the numerical serum bilirubin estimation (mg/dL) if jaundice is detected.

## 10.3 User Experience and Design Principles
The interface follows a clean, minimalist design as specified by modern UX principles. It includes appropriate medical disclaimers, emphasizing that the tool is intended for screening and research purposes only and does not replace a clinical pathological laboratory report.

## 10.4 Screen Captures of the Streamlit Interface
[INSERT_STREAMLIT_FRONTEND_SCREENSHOT_HERE]
*Figure 10.1: Main dashboard of the Streamlit-based jaundice detection application showing the image uploader and diagnostic panel.*

## 10.5 System Reliability and Feedback
The application includes error-handling mechanisms that notify the user if the uploaded image does not contain a detectable eye or if the sclera area is too small for accurate feature extraction. This prevents erroneous predictions based on low-quality data.
