# CHAPTER 5: DATASET DESCRIPTION

## 5.1 Dataset Origin and Purpose
The dataset used in this research was curated specifically for the diagnostic screening of hyperbilirubinemia. It consists of high-quality digital images of the human eye paired with clinically validated laboratory serum bilirubin values. The dataset was collected at the **St Aloysius AI Research Lab** in collaboration with clinical partners.

## 5.2 Dataset Statistics
The primary dataset utilized for the final model training and evaluation includes **177 original eye images**. These images are linked to detailed patient records containing biochemical and demographic data.

### 5.2.1 Demographic Distribution
The demographic breakdown of the study participants is as follows:
-   **Total Samples**: 177
-   **Male Patients**: 144
-   **Female Patients**: 33

This distribution reflects a higher prevalence of jaundice-related hospital admissions among the male demographic in the collection period, leading to a gender imbalance that is addressed during model training via stratifed sampling and balanced class weighting.

### 5.2.2 Bilirubin Range and Distribution
The serum bilirubin values in the dataset range from normal physiological levels (< 1.2 mg/dL) to severe pathological levels (> 15.0 mg/dL). For the binary classification task, a clinical threshold of **2.0 mg/dL** was used to define the "Jaundice" class.
-   **Normal (Normal):** Bilirubin $\leq$ 2.0 mg/dL
-   **Jaundice (Positive):** Bilirubin > 2.0 mg/dL

## 5.3 Medical Attributes
In addition to image data, the dataset includes several medical variables that serve as ground truth or contextual validation:
-   **Gender**: Boolean (M/F)
-   **Age**: Continuous variable.
-   **Blood Group**: Categorical.
-   **Clinical Manifestations**: Qualitative observations (e.g., degree of visible icterus).
-   **Serum Bilirubin Level**: Continuous target variable for regression (mg/dL).

## 5.4 Challenges and Considerations
### 5.4.1 Multiple Images per Patient
Several patients contributed multiple images to the dataset. While this increases the total sample count, it introduces dependency between samples (within-subject correlation). To prevent data leakage, model evaluation was performed while ensuring that images from the same patient did not appear in both the training and testing sets simultaneously in the patient-level analysis.

### 5.4.2 Lighting and Environmental Variations
Images were captured in varied clinical environments. Although this introduced noise (specular reflections and shadow), it was necessary to ensure the model's robustness to real-world ambient lighting conditions.

## 5.5 Preprocessing and Labeling
Every image was manually annotated for the U-Net segmentation task, creating pixel-level ground truth masks. For classification and regression, the serum bilirubin values obtained from clinical blood tests within 24 hours of image acquisition were treated as the absolute gold standard ground truth.
