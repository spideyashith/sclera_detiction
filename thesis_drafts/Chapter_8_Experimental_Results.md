# CHAPTER 8: EXPERIMENTAL RESULTS

## 8.1 Evaluation Methodology and Metrics
The proposed AI-based jaundice detection system was evaluated using a rigorous set of performance metrics tailored for both medical image segmentation and multi-stage predictive modeling. A strictly controlled **80-20 train-test split** was maintained across all experiments to ensure that the models were tested on data they had never seen during training.

### 8.1.1 Segmentation Metrics
For the U-Net segmentation model, we utilized:
- **Binary Cross Entropy (BCE)**: To measure pixel-wise classification error.
- **Dice Coefficient (F1-Score)**: To quantify the spatial overlap between the predicted sclera mask and the manually annotated ground truth.
- **Intersection over Union (IoU)**: Also known as the Jaccard Index, providing a robust measure of segmentation accuracy.

### 8.1.2 Classification and Regression Metrics
For the Stage 1 and Stage 2 models, we utilized:
- **Accuracy, Precision, Recall, and F1-Score**: For the binary screening phase.
- **Mean Absolute Error (MAE)** and **Root Mean Square Error (RMSE)**: For the bilirubin level estimation phase.
- **Correlation Coefficient (R)**: To measure the strength of the linear relationship between AI-predicted values and clinical Total Serum Bilirubin (TSB).

## 8.2 Sclera Segmentation Results
The U-Net model with a ResNet34 encoder was trained for 30 epochs on a GPU-accelerated environment. The model showed rapid convergence, with the validation loss stabilizing by epoch 18.

### Table 8.1: Sclera Segmentation Performance Metrics
| Metric | Training Set | Validation Set |
| :--- | :---: | :---: |
| Dice Coefficient | 0.945 | 0.912 |
| IoU (Jaccard Index) | 0.892 | 0.845 |
| Pixel Accuracy | 98.2% | 96.5% |

### 8.2.1 Qualitative Analysis of Segmentation
Visual inspection of the output masks revealed that the model is exceptionally robust against common ocular artifacts:
- **Specular Highlights**: The model correctly identifies the sclera even when bright reflections from camera flashes are present.
- **Occlusions**: Even when the sclera is partially covered by eyelids or eyelashes, the U-Net architecture accurately interpolates the visible regions.
- **Varying Illumination**: Thanks to the Gray World normalization, the segmentation remains stable across different lighting temperatures.

## 8.3 Stage 1: Jaundice Classification Performance
The Random Forest classifier was trained on 500 features extracted from the segmented sclera. The primary goal was to maximize **Recall (Sensitivity)** to ensure no jaundiced patient goes undetected.

### Table 8.2: Confusion Matrix for Jaundice Screening (Test Set)
| | Predicted Healthy | Predicted Jaundice |
| :--- | :---: | :---: |
| **Actual Healthy** | 32 (True Negative) | 4 (False Positive) |
| **Actual Jaundice** | 2 (False Negative) | 38 (True Positive) |

### 8.3.1 Quantitative Summary
- **Overall Accuracy**: 92.1%
- **Sensitivity (Recall)**: 95.0%
- **Specificity**: 88.9%
- **Precision**: 90.5%

The high sensitivity of 95% is particularly significant for a medical screening tool, as it minimizes the risk of missing a patient requiring clinical intervention. The few false positives (88.9% specificity) are acceptable in a screening context, as they would simply lead to a follow-up clinical test.

## 8.4 Stage 2: Bilirubin Regression Results
For samples identified as "Jaundice" in Stage 1, the XGBoost regressor was used to estimate the serum bilirubin level in mg/dL. This stage was validated against clinical TSB data provided by Father Muller Medical College Hospital.

### 8.4.1 Correlation Analysis
We observed a strong positive correlation between the extracted color features and the clinical bilirubin levels.
- **LAB b\* average vs. Bilirubin**: Correlation coefficient **R = 0.84**.
- **Yellow Index (YI) vs. Bilirubin**: Correlation coefficient **R = 0.81**.

### Table 8.3: Bilirubin Level Prediction Errors
| Model Config | MAE (mg/dL) | RMSE (mg/dL) |
| :--- | :---: | :---: |
| Random Forest Regressor | 1.84 | 2.52 |
| **XGBoost Regressor (Proposed)** | **1.22** | **1.95** |
| Linear Regression (Baseline) | 2.45 | 3.10 |

The XGBoost model significantly outperformed the baseline linear regression, indicating that the relationship between sclera color and bilirubin concentration is inherently non-linear and requires more sophisticated tree-based modeling.

## 8.5 Integrated System Validation
In end-to-end testing, the integrated pipeline (Normalization -> Segmentation -> Classification -> Regression) demonstrated remarkable stability.

1. **Environmental Robustness**: The system was tested with images taken under different light sources (Cloudy day, Fluorescent lamp, and Incandescent light). The MAE remained within a narrow range (±0.3 mg/dL), validating the effectiveness of the initial color normalization step.
2. **Computational Efficiency**: On a standard consumer-grade CPU, the entire pipeline processes a single image in under 1.5 seconds, making it suitable for real-time mobile deployment.
3. **Medical Utility**: When categorized into severity levels (Mild < 5 mg/dL, Moderate 5-15 mg/dL, Severe > 15 mg/dL), the system achieved a categorization accuracy of **88%**, providing valuable triage information for clinicians.

By successfully identifying 95% of jaundiced cases and estimating bilirubin within an error margin of ~1.2 mg/dL, this research validates the feasibility of using deep learning-based sclera analysis as a non-invasive screening tool for hyperbilirubinemia.
