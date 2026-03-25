# CHAPTER 6: METHODOLOGY

## 6.1 Color Normalization: The Gray World Algorithm
The **Gray World Algorithm** is implemented as the first preprocessing step to stabilize the color balance. It operates on the assumption that the average reflectance of a scene under achromatic lighting is gray. This is crucial for medical diagnostics where different camera sensors and light sources (e.g., LED, fluorescent, sunlight) can introduce significant color casts.

For each color channel (R, G, B), the average intensities ($\mu_R, \mu_G, \mu_B$) are calculated across the entire image. The global average intensity is then computed as the mean of these three values:
\[ \mu_{gray} = \frac{\mu_R + \mu_G + \mu_B}{3} \]

A scaling factor is applied to each channel to adjust the balance toward the global average:
\[ C_{normalized} = C_{original} \times \left( \frac{\mu_{gray}}{\mu_C} \right) \]

This process helps in mitigating the "yellow shift" caused by warm indoor lighting, which could otherwise lead to false positives in jaundice detection. It ensures that the color features subsequently extracted purely reflect the physiological state of the sclera rather than environmental artifacts.

## 6.2 Sclera Segmentation using U-Net
The core of the detection pipeline is based on a Deep Learning **U-Net** architecture. Specifically, we utilize an improved variant with a **ResNet34 encoder** pretrained on the ImageNet dataset. This transfer learning approach allows the model to leverage low-level features (like edges and textures) learned from millions of natural images.

### 6.2.1 U-Net Architecture and Design
U-Net is a convolutional neural network designed for fast and precise image segmentation. It consists of two main parts:
1. **Contracting Path (Encoder)**: This part reduces the spatial dimensions of the input image while increasing the feature depth. We use ResNet34, which consists of residual blocks that prevent the vanishing gradient problem in deep networks. Each block captures increasingly abstract features, starting from simple edges to complex shapes like the curvature of the eye.
2. **Expansive Path (Decoder)**: This part recovers the spatial resolution of the feature maps through a series of upsampling (transpose convolution) layers. It effectively translates the high-level semantic information back into a pixel-level mask.
3. **Skip Connections**: A defining feature of U-Net is the presence of skip connections that bridge the encoder and decoder at each resolution level. These connections merge the high-resolution features from the encoder directly into the decoder. This ensures that fine-grained spatial information, such as the precise boundary of the sclera near the iris and eyelids, is preserved during the upsampling process.

### 6.2.2 Training Strategy and Hyperparameters
To ensure robust segmentation performance, the model was trained with the following configurations:
- **Image Size**: All input images were resized to $256 \times 256$ pixels to balance computational efficiency with the required spatial detail.
- **Optimizer**: The **Adam optimizer** was chosen for its adaptive learning rate capabilities, with an initial learning rate of $1 \times 10^{-4}$.
- **Loss Function**: We employed a combined loss strategy. While **BCEWithLogitsLoss** handles pixel-wise classification, we also monitored the **Dice Coefficient** to ensure overlap between the predicted and ground truth masks was maximized.
- **Data Augmentation**: To improve generalization, we applied random rotations, horizontal flips, and color jittering. This prevents the model from overfitting to specific camera angles or lighting conditions present in the training set.
- **Epochs**: The model was trained for 20-30 epochs, with early stopping based on validation loss to prevent overfitting.

## 6.3 Color Space and Feature Engineering
Following successful segmentation, the system extracts a rich set of statistical features from the isolated sclera region. Feature engineering is a critical step that bridges the gap between raw pixel data and clinical biomarkers.

### 6.3.1 RGB and HSV Feature Extraction
- **RGB Analysis**: The RGB color space provides a direct measure of the red, green, and blue intensities. Physiologically, bilirubin has a strong absorption peak in the blue part of the spectrum (~450 nm). This leads to a characteristic decrease in the blue channel intensity relative to the red channel in jaundiced eyes.
- **HSV Analysis**: Moving beyond RGB, the HSV (Hue, Saturation, Value) space allows for more stable color quantification. Hue, in particular, is a robust marker of 'yellowness' that is relatively invariant to changes in illumination intensity. We extract the mean and standard deviation of the Hue channel within the segmented sclera.

### 6.3.2 CIE LAB Color Space and Perceptual Uniformity
The **CIELAB** color space is designed to be perceptually uniform, meaning the numerical difference between two colors corresponds to the difference perceived by the human eye.
- **L***: Represents lightness, which we use to filter out extremely dark or bright specular reflection regions.
- **a***: Represents the green-red axis.
- **b*** (Yellow-Blue axis): This is the most critical feature for this research. Positive values on the b* axis directly quantify the degree of yellowing. We found that the mean b* value is the single strongest predictor of serum bilirubin levels.

### 6.3.3 Advanced Statistical Features
In addition to simple means, we extract higher-order statistics to capture the distribution of colors:
- **Standard Deviation**: Provides information about the uniformity of the yellowing.
- **Skewness and Kurtosis**: Help identify whether the discoloration is localized or spread across the entire sclera.
- **Yellow Index (YI)**: A custom-engineered feature defined as \( (R - B) / (R + B + G) \). This ratio highlights the dominance of red/yellow over blue, acting as a normalized marker for jaundice.

## 6.4 Stage 1: Random Forest Classification for Screening
The first stage of our predictive modeling is a binary classifier designed to distinguish between "Healthy" and "Jaundice" cases. We use a **Random Forest (RF)** model for this task.

- **Architecture**: The forest consists of 500 independent decision trees. Each tree is trained on a random bootstrap sample of the data, and each node split considers a random subset of features. This ensemble approach provides high accuracy and prevents the model from being overly sensitive to any single feature.
- **Handling Class Imbalance**: Clinical datasets often have an imbalance between healthy and diseased samples. We address this using the `class_weight='balanced'` parameter, which automatically adjusts the weights inversely proportional to class frequencies.
- **Feature Importance**: One of the primary advantages of Random Forest is its ability to quantify the contribution of each feature to the final decision. We utilize **Gini Importance** to rank our extracted features, confirming that the b* channel and Yellow Index are indeed the most discriminative.

## 6.5 Stage 2: Bilirubin Level Regression with XGBoost
For images identified as jaundiced in Stage 1, the system proceeds to Stage 2: estimating the actual concentration of serum bilirubin in mg/dL. This is a regression problem, for which we employ **XGBoost (Extreme Gradient Boosting)**.

- **Why XGBoost?**: Gradient boosting machines are known for their state-of-the-art performance on tabular data. Unlike simple linear regression, XGBoost can capture complex non-linear relationships between color features and bilirubin levels.
- **Training Logic**: The regressor is trained exclusively on samples with known TSB (Total Serum Bilirubin) values. By focusing only on the "Jaundice" class, the model learns the subtle gradients in color that correspond to increasing disease severity (e.g., distinguishing between mild jaundice at 5 mg/dL and severe jaundice at 20 mg/dL).
- **Hyperparameter Optimization**: We tuned the model using a grid search over the number of estimators (300-500), learning rate (0.01-0.1), and tree depth (3-6). A lower learning rate with more estimators was found to provide the best generalization on the test set.
## 6.6 Formal Mathematical Framework
In this section, we define the mathematical foundation used for image processing and bilirubin estimation.

### 6.6.1 Color Normalization (Gray World)
To achieve consistency across varying light sources ($L$), each pixel color channel $C \in \{R, G, B\}$ is transformed using a global scaling factor $K_C$:
\[ C_{norm}(x,y) = K_C \cdot C_{orig}(x,y) \]
Where the scaling factor is derived from the mean intensities $\mu$:
\[ K_C = \frac{\sum_{i \in \{R,G,B\}} \mu_i}{3 \cdot \mu_C} \]

### 6.6.2 Feature Engineering (Yellow Index)
The primary clinical biomarker derived from the segmented sclera is the **Yellow Index (YI)**, which characterizes the dominance of long-wavelength (red/yellow) reflectance over short-wavelength (blue) absorption by bilirubin:
\[ YI = \frac{R - B}{R + G + B} \]
Where $R$ and $B$ are the mean intensities of the red and blue channels within the segmented ROI.

### 6.6.3 Bilirubin Level Estimation Model
The continuous estimation of Total Serum Bilirubin (TSB) in $mg/dL$ is modeled as a non-linear function $F(\cdot)$ learned by the gradient-boosted ensemble:
\[ \text{TSB}_{predicted} = \sum_{m=1}^{M} f_m(v_{feature}) \]
Where $v_{feature}$ is the 10-dimensional input vector:
\[ v = [ \bar{R}, \bar{G}, \bar{B}, \bar{H}, \bar{S}, \bar{V}, \bar{L^*}, \bar{a^*}, \bar{b^*}, YI ] \]
And $f_m$ represents individual decision trees optimized to minimize the Mean Square Error (MSE):
\[ \mathcal{L} = \frac{1}{N} \sum_{i=1}^{N} (\text{TSB}_{actual} - \text{TSB}_{pred})^2 \]


## 6.7 Architectural Workflow Overview
[INSERT_WORKFLOW_DIAGRAM_HERE]
*Figure 6.1: Operational Workflow showcasing the multi-stage diagnostic process from image input to patient reporting.*

Through this multi-feature regression, the system captures subtle ocular chromatic shifts that correlate with high-precision clinical lab reports.
