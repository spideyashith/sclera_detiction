# CHAPTER 7: IMPLEMENTATION

## 7.1 Development Environment and Tools
The implementation of the jaundice detection system was conducted using a modern Python-based stack, optimized for machine learning and computer vision.

### 7.1.1 Programming Language and Core Libraries
-   **Python 3.12**: The primary language for development, offering a rich ecosystem of IA and scientific libraries.
-   **OpenCV (Open Source Computer Vision Library)**: Used for image loading, color space conversions (RGB, HSV, LAB), and basic image processing tasks.
-   **PyTorch**: The deep learning framework used to implement, train, and deploy the U-Net segmentation model.
-   **Segmentation Models PyTorch (SMP)**: A high-level library used to integrate the ResNet34 encoder with the U-Net architecture.
-   **Scikit-Learn**: Utilized for feature scaling (StandardScaler), model evaluation metrics, and the Random Forest classification algorithm.
-   **XGBoost**: Employed for high-performance gradient boosting in the bilirubin regression stage.
-   **Pandas and NumPy**: Essential for structured data manipulation and numerical computations.

## 7.2 Hardware Specifications
-   **Processor**: Intel Core i5 / i7 multi-core processor.
-   **Graphics Processing Unit (GPU)**: NVIDIA GPU with CUDA support (used for accelerating U-Net training).
-   **Memory**: 8GB / 16GB RAM.
-   **Storage**: 512GB SSD for fast dataset access.

## 7.3 Modular Software Implementation
The software is organized into several functional scripts, ensuring modularity and maintainability:

### 7.3.1 Segmentation Module (`train_sclera_segmentation.py`)
This script handles the training of the U-Net model. It includes:
-   A custom `ScleraDataset` class for loading images and masks.
-   Data augmentation using `Albumentations` (Resize to 256x256).
-   Training loop with `BCEWithLogitsLoss` and model serialization to `sclera_segmentation_model.pth`.

### 7.3.2 Classification Module (`train_jaundice_classifier.py`)
This script implements the Stage 1 model:
-   Loads features from `features_dataset.csv`.
-   Applies `StandardScaler` to the feature set.
-   Trains a `Random Forest Classifier` with 300 estimators and `class_weight='balanced'`.

### 7.3.3 Regression Module (`train_bilirubin_regressor.py`)
This script focuses on Stage 2:
-   Filters the dataset to include only jaundiced samples (Bilirubin > 2.0).
-   Trains an `XGBRegressor` to estimate continuous bilirubin values.
-   Outputs the finalized `bilirubin_regressor.pkl`.

### 7.3.4 Integrated Pipeline (`predict_jaundice_pipeline.py`)
This is the production-ready script that ties all components together. It accepts a raw eye image and performs the full sequence:
-   `gray_world_normalization`: Preprocessing.
-   `segment_sclera`: U-Net inference.
-   `extract_features`: Statistical computation.
-   `Classification & Regression`: Dual-stage ML inference.

## 7.4 Deployment with Streamlit
The final system is deployed as a web application using **Streamlit** (`frontend_app.py`). The interface allows users to upload an image via a file uploader, triggers the inference pipeline, and displays:
1.  The original eye image.
2.  The segmented sclera (highlighted via a green overlay).
3.  Jaundice probability and prediction status.
4.  The estimated bilirubin level (if jaundice is detected).
