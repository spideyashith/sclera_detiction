# CHAPTER 2: LITERATURE REVIEW

## 2.1 Introduction to Medical Image Analysis
Medical imaging has undergone a paradigm shift with the integration of Artificial Intelligence. Computer-Aided Diagnosis (CAD) systems are now capable of assisting clinicians in interpreting complex medical data with high precision. Specifically, the analysis of exterior physiological markers—such as the eye, skin, and nails—has gained prominence due to the non-invasive nature of data acquisition. The evolution of medical imaging from simple X-rays to advanced MRI and CT scans paved the way for image processing in diagnostics. However, for many conditions like jaundice, the focus is now shifting toward accessible, non-contact methods that can be used in rural or remote areas without high-cost infrastructure.

## 2.2 Evolution of Jaundice Detection Methods
Historically, jaundice was assessed through visual inspection by clinicians. However, this is highly subjective and prone to inter-observer variability. The Kramer Scale, for instance, provides a visual estimate of jaundice progression from head to toe but lacks the numerical precision required for clinical monitoring. Factors such as ambient lighting, the experience of the healthcare provider, and the patient's skin tone significantly impact the accuracy of visual bedside assessments.

### 2.2.1 Transcutaneous Bilirubinometry (TcB)
The first wave of non-invasive tools involved transcutaneous bilirubinometers, which use multi-wavelength spectral reflection to estimate bilirubin in the subcutaneous tissue. These devices function by measuring the light reflection of specific wavelengths (usually blue and green) after they penetrate the skin. While effective for neonates, these devices are expensive and their accuracy can be influenced by skin melanin and thickness. Furthermore, they require physical contact, which might not be ideal in sensitive cases or infection control environments.

### 2.2.2 Smartphone-Based Approaches
With the ubiquity of high-resolution smartphone cameras, researchers began exploring image-based bilirubin estimation. Early attempts utilized simple RGB analysis of skin patches. However, variations in skin pigmentation across different ethnicities posed a significant challenge to the generalizability of these models. Researchers soon realized that the human eye, specifically the sclera, might provide a more stable and universal diagnostic marker than the skin.

## 2.3 Color Space Analysis in Medical Diagnostics
Color space selection is a critical factor in jaundice detection. Bilirubin accumulation manifests as a yellow pigment, which is essentially a shift in the chromatic properties of the tissue.
- **RGB Color Space**: While standard, RGB is highly dependent on lighting conditions and camera sensor characteristics. It lacks a clear separation between luminance and chrominance.
- **HSV/HSL Color Space**: By separating Hue (H), Saturation (S), and Value (V), researchers can better isolate the 'yellow' component of jaundice, regardless of brightness.
- **CIE LAB Color Space**: This is a device-independent color space where 'L' represents lightness, 'a' represents the green-red axis, and 'b' represents the blue-yellow axis. In the context of jaundice detection, the 'b' channel is particularly significant as it directly correlates with the intensity of yellowing in the sclera.

## 2.4 Sclera-Based Analysis
The sclera is considered a superior site for bilirubin assessment because it lacks the confounding melanin found in skin. However, isolating the sclera is a non-trivial task due to the presence of eyelashes, eyelids, blood vessels, and varying lighting conditions. The sclera acts as a white reflector, making any yellowing highly apparent compared to other ocular structures like the iris or pupil.

### 2.4.1 Classical Image Processing Techniques
Initial research in sclera segmentation relied on color thresholding (e.g., HSV-based masks) and morphological operations. While computationally lightweight, these methods often fail in real-world scenarios where shadows and highlights distort the color distribution. Features such as Gabor filters and Local Binary Patterns (LBP) were often used to distinguish the texture of the sclera from the surrounding skin, but these were sensitive to noise.

### 2.4.2 Deep Learning in Segmentation
The emergence of Convolutional Neural Networks (CNNs), particularly the U-Net architecture, revolutionized medical image segmentation. Ronneberger et al. (2015) introduced U-Net as a symmetric encoder-decoder network that captures both context and spatial localization. The skip connections in U-Net allow the network to retain high-resolution spatial information, which is essential for defining the precise boundaries of the sclera. 

Variants like ScleraSegNet have since been proposed specifically for ocular biometric and diagnostic tasks. These models often incorporate attention mechanisms to focus on relevant pixels while ignoring artifacts like specular reflections from the cornea. Sclera segmentation is now considered a foundational step in any ocular diagnostic pipeline.

## 2.5 Machine Learning Models in Healthcare
Machine learning models like Support Vector Machines (SVM), Random Forest (RF), and XGBoost have been widely applied to structured medical data.
- **Random Forest**: As an ensemble of decision trees, RF is highly robust to noise and provides a measure of feature importance, which is crucial for interpretability in medical contexts. It excels in handling high-dimensional feature sets extracted from images.
- **XGBoost**: Known for its efficiency and predictive power, XGBoost is particularly effective for regression tasks, such as estimating continuous bilirubin levels from extracted color features. Its gradient boosting framework allows it to correct residual errors recursively.
- **Ensemble Learning**: Recent research suggests that combining multiple models (stacking or voting) can provide more stable predictions than any single model, which is vital for clinical safety.

## 2.6 Comparative Study of Existing Frameworks
Several research groups have proposed end-to-end systems for jaundice screening.
- **BiliScreen (Mariakakis et al., 2017)**: One of the most prominent works, BiliScreen uses a smartphone camera and a paper color calibration card to normalize images before analyzing the sclera. Their results showed a strong correlation (0.89) with clinical TSB tests.
- **ScleraColor (Gupta et al., 2020)**: This framework focused on adult jaundice and utilized specialized illumination to enhance the yellow pigment. However, the requirement for specialized hardware limits its widespread use.
- **U-Net + Regressor (Ahmed et al., 2022)**: Similar to this project, their approach utilized U-Net for segmentation but relied on a simpler linear regression model for bilirubin estimation. Our work expands on this by using advanced non-linear regressors like XGBoost.

## 2.7 Challenges in Global Deployment
Implementing AI-based jaundice detection on a global scale involves several technical and ethical hurdles.
1. **Ethical Considerations**: Data privacy is paramount in medical AI. Systems must ensure that patient images are handled securely and in compliance with regulations like GDPR or HIPAA.
2. **Device Heterogeneity**: Image quality varies significantly across smartphones. A robust system must include sophisticated color normalization techniques (like the Gray World algorithm) to ensure consistency across different hardware.
3. **Clinical Validation**: AI models must undergo rigorous clinical trials to ensure they are safe for screening purposes. They are intended to complement, not replace, professional medical diagnosis.

## 2.8 Summary of Related Works
Recent studies have showcased the potential of combining deep learning for feature extraction with classical ML for classification. For example:
- **Esteva et al. (2019)** demonstrated that deep learning-enabled medical vision can match or exceed dermatologists' performance in certain tasks.
- **Wang et al. (2019)** showed that attention-based U-Net models can achieve high accuracy in sclera segmentation even under challenging lighting.
- **Chen et al. (2021)** proposed a multi-site diagnostic framework (skin + eye) that significantly improved jaundice detection sensitivity.

## 2.9 Research Gap
Despite significant progress, several gaps remain:
1.  **Dataset Diversity**: Many studies rely on controlled clinical datasets that do not reflect the diversity of real-world, home-based environments.
2.  **Explainability**: There is a need for systems that provide visual feedback (like segmented masks) to help clinicians understand the rationale behind a prediction.
3.  **End-to-End Pipelines**: Few systems provide a seamless bridge from raw image capture to real-time server-side processing and instant feedback to the user.

This project addresses these gaps by implementing a robust U-Net-based pipeline optimized for adult sclera analysis, incorporating color normalization and interpretable feature engineering, and deploying it through an interactive web-based interface.
