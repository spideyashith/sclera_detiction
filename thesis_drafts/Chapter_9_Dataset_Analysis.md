# CHAPTER 9: DATASET ANALYSIS

## 9.1 Gender-Based Jaundice Analysis
A detailed analysis was performed to understand the distribution of jaundice across genders within the study population.
-   **Male Jaundice Percentage**: Approximately **45-50%** of the male samples exhibited hyperbilirubinemia.
-   **Female Jaundice Percentage**: Jaundice was present in a similar proportion of female samples, though the total sample size for females was smaller (33 vs 144).
This analysis confirms that the color biomarkers (Yellow Index, LAB 'b') are globally applicable across genders, provided that the segmentation model effectively handles varying eye shapes.

## 9.2 The Impact of Multiple Images per Patient
As noted in Chapter 5, the dataset contains several images per patient.
-   **Observations**: Multiple images from the same patient under different lighting confirmed that the system is robust to environmental changes.
-   **In-patient Variance**: We observed that while the bilirubin ground truth remains constant for a single patient on a given day, the extracted features (e.g., Mean R, G, B) exhibited slight variations due to shadows and reflections.
-   **System Performance**: The use of Stage 1 and Stage 2 models together helped in "smoothing" these variations, as the ensemble nature of Random Forest and XGBoost is inherently robust to minor outliers.

## 9.3 Feature Correlation Analysis
Correlation matrices between extracted features and serum bilirubin levels were computed.
-   **Strong Positive Correlations**:
    -   Bilirubin vs. Yellow Index ($R - B$): high positive correlation.
    -   Bilirubin vs. LAB 'b' component: high positive correlation.
-   **Negative Correlations**:
    -   Bilirubin vs. Mean Blue (B): high negative correlation (due to light absorption).
    -   Bilirubin vs. Mean Hue (H): inverse relationship observed as the hue shifts towards the yellow spectrum.

This quantitative analysis validates our feature engineering approach, confirming that the selected color spaces are medically relevant for non-invasive icterus assessment.
