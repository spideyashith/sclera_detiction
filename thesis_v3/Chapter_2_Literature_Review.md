# CHAPTER 2

# LITERATURE REVIEW

## 2.1 INTRODUCTION

The transformation of medical diagnostics through Artificial Intelligence (AI) and Machine Learning (ML) has been a significant theme in recent biomedical engineering research. The ability of computer vision algorithms to perceive subtle details that are invisible to the human eye, such as minute color shifts or texture variations in physiological tissues, has paved the way for non-invasive, cost-effective, and rapid screening tools. This chapter provides a comprehensive review of the scholarly work related to AI-driven health monitoring, specifically focusing on jaundice detection, the broader landscape of AI in healthcare, and advanced neural network architectures like Long Short-Term Memory (LSTM) networks and Convolutional Neural Networks (CNNs).

## 2.2 AI IN HEALTHCARE: AN EVOLVING LANDSCAPE

Artificial Intelligence has transitioned from being a purely experimental tool to a core component of modern clinical decision-support systems. In healthcare, AI applications range from automated radiology report generation to the prediction of patient outcomes using Electronic Health Records (EHR). 

Recent studies by [Ref 1] highlight the efficacy of deep learning in identifying pathologies from high-resolution medical imagery. For instance, the use of AI in detecting diabetic retinopathy, skin cancer, and cardiovascular abnormalities has shown performance parity with, and in some cases superiority to, human specialists. The foundational work in AI-driven healthcare emphasizes the importance of data quality, model interpretability, and the robustness of algorithms across diverse patient populations.

## 2.3 RECENT TRENDS IN DEMENTIA DETECTION AND NEUROIMAGING

While the primary focus of this project is jaundice, it is essential to acknowledge the parallels in AI-based diagnostics for other chronic conditions, such as dementia. Early detection of neurodegenerative diseases, including Alzheimer’s, has leveraged machine learning to analyze brain MRI scans and cognitive performance data. 

[Ref 12] and [Ref 15] discuss how deep learning models, particularly 3D CNNs and Recurrent Neural Networks (RNNs), are used to identify hippocampal atrophy and other biomarkers of cognitive decline. These methodologies share a common technical foundation with jaundice detection: the extraction of high-level features from physiological data and the use of probabilistic classifiers to provide early warnings before significant clinical symptoms manifest. The cross-pollination of techniques from dementia research, such as attention mechanisms and multi-modal data fusion, offers valuable insights for enhancing the accuracy of color-based jaundice screening.

## 2.4 LONG SHORT-TERM MEMORY (LSTM) NETWORKS IN MEDICAL MONITORING

In the realm of temporal medical data—such as ECG signals, patient vital sign monitoring, or the progression of a disease over time—Recurrent Neural Networks, specifically Long Short-Term Memory (LSTM) networks, have proven indispensable. LSTMs are designed to handle the vanishing gradient problem in standard RNNs, allowing them to learn long-term dependencies in sequential data.

[Ref 18] explores the use of LSTMs in real-time medical monitoring systems, where they can predict critical events (e.g., sepsis or cardiac arrest) by analyzing historical patient data. In the context of jaundice, although the current study focuses on static image analysis, LSTM networks could potentially be employed in future work to track the progression of bilirubin levels over multiple days, providing a longitudinal view of the patient’s recovery or deterioration. The integration of LSTMs with medical monitoring systems [Ref 20] represents a significant advancement in proactive patient care.

## 2.5 IMAGE-BASED JAUNDICE DETECTION: STATE OF THE ART

The use of digital imaging to analyze the color of the sclera for jaundice detection has been explored by several researchers. Key methodologies in this field include:

-   **Colorimetric Analysis**: Early works utilized standard RGB color space analysis to calculate the "yellowness" of the sclera. However, RGB is highly sensitive to lighting variations.
-   **Multi-Spectrum Imaging**: Some researchers have used specialized hardware to capture images across different wavelengths, providing a more accurate estimation of bilirubin absorption.
-   **Deep Learning Segmentation**: The use of CNNs, particularly U-Net, for isolating the sclera region has significantly improved the robustness of image-based diagnostics by excluding eyelids, eyelashes, and reflections [Ref 25].

Studies by [Ref 8] have shown that converting images to the CIELAB color space (L*a*b*) offers better perceptual uniformity, where the 'b*' channel (yellow-blue axis) serves as a strong indicator of bilirubin presence. Furthermore, the combination of hand-crafted color features with ensemble machine learning models like Random Forest has yielded high classification accuracies in clinical trials.

## 2.6 ANALYSIS OF RELATED WORKS AND RESEARCH GAPS

Despite the progress made in non-invasive jaundice detection, several research gaps remain:

1.  **Lighting Sensitivity**: Many existing systems fail in low-light or variable-temperature lighting conditions, where the color of the sclera is artificially tinted.
2.  **Skin Tone Variation**: While the sclera is generally white, the surrounding skin tone can affect the camera’s white-balance and auto-exposure algorithms, indirectly influencing the sclera's pixel values.
3.  **Lack of Large-Scale Validated Datasets**: Most studies are conducted on small, localized datasets with limited ethnic diversity, leading to models that may not generalize globally.
4.  **Hardware Dependency**: Many systems require high-end cameras or specific color-calibration cards, which limits their usability in everyday smartphone-based screening.

This project addresses these gaps by implementing a robust segmentation model that is less sensitive to background noise and exploring robust feature extraction techniques that integrate data from multiple color spaces.

## 2.7 LITERATURE COMPARISON

The following table summarizes the key contributions and methodologies of relevant studies in the field:

| Author | Year | Methodology | Focus Area | Accuracy / Metric |
| :--- | :--- | :--- | :--- | :--- |
| Taylor et al. [Ref 2] | 2017 | BiliCam App | Neonatal Jaundice | 0.85 Correlation |
| Padidar et al. [Ref 5] | 2019 | CIELAB Color Analysis | Adult Jaundice | 92% Accuracy |
| Marvasti et al. [Ref 9]| 2021 | U-Net + CNN | Sclera Segmentation | 0.94 Dice Coeff |
| Chen et al. [Ref 14] | 2022 | LSTM for Vitals | Patient Monitoring | 0.89 F1-Score |
| Zhou et al. [Ref 21] | 2023 | Ensemble Learning | Multi-feature Analysis | 94.5% Accuracy |

## 2.8 METHODOLOGY COMPARISON

When comparing methodologies for jaundice detection, three main approaches emerge:

1.  **Manual Color Patch Comparison**: Involves a physical card placed near the eye. While simple, it is prone to human error and lighting issues.
2.  **Semi-Automated Digital Analysis**: Requires a user to manually select the sclera region. This is time-consuming and inconsistent.
3.  **Fully Automated AI Pipelines (Proposed)**: Uses deep learning to automatically segment, extract features, and classify. This approach offers the highest scalability and consistency, though it requires significant computational resources for training.

In conclusion, the literature suggests that a robust, non-invasive system for jaundice detection must combine precise anatomical segmentation with multi-space color analysis and advanced machine learning classifiers to overcome the challenges of environmental variability.

---
