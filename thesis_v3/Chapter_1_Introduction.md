# CHAPTER 1

# INTRODUCTION

## 1.1 OVERVIEW / BACKGROUND

Jaundice, or icterus, is a clinical condition characterized by a yellowish discoloration of the skin, mucous membranes, and the sclera (the white part of the eye). This discoloration is caused by the accumulation of bilirubin, a yellow-tinged pigment formed during the normal breakdown of red blood cells. Bilirubin is typically processed by the liver and excreted through bile. However, when the production of bilirubin exceeds the liver's capacity to process it, or when there is an obstruction in the biliary system, bilirubin levels in the blood rise, leading to hyperbilirubinemia and the visible signs of jaundice.

In the context of neonatal health, jaundice is one of the most common reasons for hospital readmission in the first week of life. While most cases are physiological and resolve spontaneously, severe hyperbilirubinemia can lead to irreversible brain damage, a condition known as kernicterus, or even neonatal death if left untreated. In adults, jaundice often indicates underlying hepatobiliary or hemolytic disorders, including hepatitis, cirrhosis, gallstones, or pancreatic cancer.

The detection of jaundice is traditionally performed through visual inspection by clinicians, followed by a Serum Bilirubin (SBR) test, which requires drawing blood. While visual inspection is common, it is highly subjective and depends heavily on the clinician's experience and the ambient lighting conditions. In neonates, visual assessment is particularly unreliable across different skin tones. The SBR test, although accurate, is invasive, causes pain and stress to infants, requires specialized laboratory infrastructure, and involves a waiting period for results.

Recent advancements in computer vision and artificial intelligence (AI) have opened new avenues for non-invasive medical diagnostics. Smartphone-based imaging has the potential to provide a rapid, low-cost, and accessible screening method for jaundice. Since the sclera is one of the first areas to exhibit yellowing, analyzing its color characteristics through digital imaging can provide a proxy for bilirubin concentration. By training deep learning models to recognize the subtle color shifts in the sclera, we can develop a tool that offers immediate screening results, especially valuable in resource-limited settings where laboratory access is limited.

## 1.2 PROBLEM STATEMENT

The primary challenge in jaundice diagnosis is the reliance on invasive blood tests and the subjectivity of visual clinical assessment. Specifically:

1.  **Invasivity**: The standard SBR test requires blood samples, which is distressing for neonates and requires trained medical personnel.
2.  **Infrastructure Requirements**: Rural and underprivileged areas often lack the laboratory facilities and equipment (like spectrophotometers) needed for bilirubin analysis.
3.  **Cost and Time**: Laboratory tests are relatively expensive and time-consuming, preventing rapid bedside screening.
4.  **Subjectivity**: Visual diagnosis by human observers is prone to error and varies with environmental lighting and the observer's perception.
5.  **Neonatal Risk**: Delayed diagnosis of severe jaundice in newborns can lead to permanent neurological damage (kernicterus).

There is an urgent need for a reliable, non-invasive, and automated system that can accurately detect jaundice and estimate bilirubin levels from digital eye images, providing an accessible screening tool for both clinical and home-based use.

## 1.3 PROJECT OBJECTIVES

The overarching goal of this project is to develop and evaluate an AI-driven system for the non-invasive detection of jaundice through sclera image analysis. The specific objectives are:

1.  **Automated Sclera Segmentation**: To implement a robust deep learning model (U-Net) capable of accurately isolating the sclera region from diverse eye images, regardless of background noise or variation in facial features.
2.  **Quantitative Color Analysis**: To extract and analyze dominant color features from the segmented sclera across multiple color spaces (RGB, CIELAB, and HSV) to identify the most predictive biomarkers for jaundice.
3.  **Jaundice Classification**: To develop and train machine learning classifiers (Random Forest, XGBoost) to categorize images into 'Jaundiced' and 'Healthy' with high precision and recall.
4.  **Bilirubin Level Estimation**: To design a regression model capable of predicting quantitative serum bilirubin levels from image features, providing a numerical estimate comparable to clinical laboratory values.
5.  **Performance Evaluation and Validation**: To rigorously test the system's performance using metrics such as Accuracy, F1-score, Area Under the ROC Curve (AUC), and Mean Absolute Error (MAE), validating its effectiveness against ground-truth clinical data.

## 1.4 EXPECTED OUTCOMES

The successful completion of this project is expected to yield the following outcomes:

1.  **Accurate Segmentation Engine**: A high-performance U-Net model specialized in sclera localization, which can be reused in other ophthalmic imaging applications.
2.  **Diagnostic Classification Tool**: A reliable binary classifier that can distinguish between jaundiced and healthy individuals based on sclera color analysis.
3.  **Predictive Bilirubin Modeler**: A regression system that provides a non-invasive estimate of bilirubin concentration, serving as a rapid screening metric.
4.  **Feature Importance Identification**: Insights into which color channels and mathematical features are most indicative of bilirubin levels in the eye.
5.  **Feasibility Report**: A comprehensive analysis of the system's accuracy and its potential for deployment as a mobile-based diagnostic application.

## 1.5 ORGANIZATION DETAILS

The documentation is organized into several chapters, each detailing a critical phase of the project:

-   **Chapter 1 (Introduction)**: Provides the background, defines the problem, and outlines the objectives and expected outcomes of the study.
-   **Chapter 2 (Literature Review)**: Reviews existing research in jaundice detection, color-based diagnostics, and the application of deep learning in medical imaging, identifying current gaps and the rationale for the proposed approach.
-   **Chapter 3 (Materials, Methods and Methodology)**: Detailed description of the dataset, preprocessing techniques, the U-Net architecture for segmentation, feature extraction processes, and the machine learning models used for classification and regression. This chapter also includes relevant code implementations as per the study's framework.
-   **Chapter 4 (Results and Analysis)**: Presents the experimental results, performance metrics, visualizations of model outputs, and a comparative analysis of different modeling approaches.
-   **Chapter 5 (Conclusion and Future Work)**: Summarizes the project's findings, acknowledges the limitations of the current study, and proposes directions for future research and system improvements.
-   **References**: Lists the scholarly works and citations referenced throughout the documentation in IEEE/APA style.
-   **Appendix**: Contains the full source code used for data processing, model training, and inference.

---
