# CHAPTER 11: LIMITATIONS

## 11.1 Introduction
While the developed AI-based jaundice detection system demonstrates significant potential for non-invasive screening, several technical and physiological limitations must be acknowledged. Understanding these constraints is essential for the future refinement and clinical deployment of the tool.

## 11.2 Dataset Constraints
-   **Sample Size**: The current dataset consists of 177 primary images. While sufficient for a proof-of-concept and academic research, a much larger, multi-centric dataset is required to ensure the model's performance across diverse ethnic backgrounds and age groups.
-   **Class Imbalance**: The clinical nature of the dataset acquisition (hospital-acquired) led to a higher proportion of jaundice cases compared to normal samples. Although the model uses balanced weighting, a more naturally distributed dataset would improve real-world generalization.

## 11.3 Environmental and Lighting Sensitivity
Despite the use of the **Gray World Algorithm** for color normalization, extreme lighting variations—such as strong direct sunlight or very low-light conditions—can still introduce noise into the color features. Shadows and highlights on the curved surface of the sclera can distort the extracted RGB and LAB values, potentially leading to inaccurate bilirubin estimates.

## 11.4 Anatomical and Physiological Variations
-   **Occlusion**: Deep-set eyes, thick eyelashes, and drooping eyelids (ptosis) can obstruct the sclera, reducing the number of pixels available for feature extraction. This can lead to higher variance in statistical features.
-   **Concurrent Ocular Conditions**: Conditions such as subconjunctival hemorrhage (blood in the eye) or severe pterygium (tissue growth) can interfere with the color signature of jaundice, leading to false positives or segmentation failures.

## 11.5 Clinical Limitation
The system provides a screening estimate and not a definitive diagnostic value. It is inherently limited by the quality of the eye image and cannot currently account for other biochemical markers found in traditional liver function tests (LFTs).
