# TITLE PAGE

**AI-BASED NON-INVASIVE JAUNDICE DETECTION USING SCLERA IMAGE ANALYSIS**

<br><br>

*A Project Record Submitted to*
**ST ALOYSIUS (DEEMED TO BE UNIVERSITY), MANGALURU**
In partial fulfillment for the award of the degree of

**MASTER OF SCIENCE**
In **DATA SCIENCE**

<br><br>

*Submitted by*
**ASHITH JOSWA**
**(Reg. No: 247101)**

<br><br>

*Under the guidance of*
**DR. RUBAN S**
*Associate Professor, Dept. of Computer Science*

<br><br>

**ST ALOYSIUS (DEEMED TO BE UNIVERSITY)**
**MANGALURU - 575003**
**MARCH 2026**

---

# CERTIFICATE OF AUTHENTICATED WORK

This is to certify that the project entitled **"AI-BASED NON-INVASIVE JAUNDICE DETECTION USING SCLERA IMAGE ANALYSIS"** is a bonafide work carried out by **ASHITH JOSWA (Reg. No: 247101)**, a student of **M.Sc. Data Science**, St Aloysius (Deemed to be University), Mangaluru, in partial fulfillment of the requirements for the award of the degree of Master of Science during the academic year 2024-2026.

This work has been done under my supervision and guidance. The results embodied in this report have not been submitted to any other University or Institute for the award of any degree or diploma.

<br><br><br>

**DR. RUBAN S**  
(Project Guide)  
Associate Professor  
Dept. of Computer Science  

<br><br><br>

**DR. HEMALATHA N**  
(Head of Department)  
Dept. of Computer Science  

---

# ACKNOWLEDGEMENT

I wish to express my sincere gratitude to all those who have helped me in the successful completion of this project.

First and foremost, I thank the Almighty for His blessings and for giving me the strength to complete this work.

I express my deepest gratitude to **Rev. Fr. Praveen Martis SJ**, Vice-Chancellor, St Aloysius (Deemed to be University), for providing the necessary facilities and a conducive environment for research.

I am extremely grateful to **Dr. Hemalatha N**, Head of the Department of Computer Science, for her constant encouragement and support throughout the course.

I owe a special debt of gratitude to my guide, **Dr. Ruban S**, Associate Professor, Dept. of Computer Science, for his invaluable guidance, insightful suggestions, and constant motivation at every stage of this project. His expertise and dedication have been instrumental in shaping this work.

I also thank all the faculty members of the Department of Computer Science for their support and guidance.

My heartfelt thanks go to my family and friends for their unwavering support, patience, and encouragement during this journey.

Finally, I acknowledge the various online resources and research communities whose contributions to the field of medical imaging and deep learning have provided the foundation for this project.

**ASHITH JOSWA**

---

# DECLARATION

I, **ASHITH JOSWA**, hereby declare that the project titled **"AI-BASED NON-INVASIVE JAUNDICE DETECTION USING SCLERA IMAGE ANALYSIS"** is an original work done by me under the guidance of **Dr. Ruban S**, Associate Professor, Dept. of Computer Science, St Aloysius (Deemed to be University), Mangaluru.

I also declare that this project or any part of it has not been submitted previously to any other university or institution for the award of any degree, diploma, fellowship, or other similar titles.

<br><br><br>

**Place: Mangaluru**  
**Date: 20-03-2026**  
**ASHITH JOSWA**

---

# PROJECT PROPOSAL REPORT / SYNOPSIS

## I. Title of the Project
AI-Based Non-Invasive Jaundice Detection Using Sclera Image Analysis

## II. Statement of the Problem
Jaundice is characterized by the yellowing of the skin and eyes (sclera) due to elevated bilirubin levels in the blood. Traditional diagnosis requires invasive blood tests, which are painful For infants, time-consuming, and inaccessible in rural areas. There is a need for a non-invasive, cost-effective, and rapid screening tool that can estimate bilirubin levels or detect jaundice using simple mobile-captured images of the eye.

## III. Why this particular topic chosen?
The motivation stems from the high prevalence of neonatal jaundice and the lack of diagnostic facilities in resource-limited settings. By leveraging advanced computer vision and deep learning techniques, we can transform a standard smartphone into a diagnostic tool, enabling early detection and timely medical intervention without the need for specialized laboratory equipment.

## IV. Objective and Scope
- To develop a deep learning model (U-Net) for precise segmentation of the sclera from eye images.
- To extract color features (RGB, LAB, HSV) from the segmented sclera region.
- To build a classification model to detect the presence of jaundice.
- To develop a regression model to estimate the serum bilirubin levels from image features.
- The scope includes processing a dataset of sclera images and validating the results against clinical blood test data.

## V. Methodology
The project follows a multi-stage pipeline:
1. Data Collection and Preprocessing (Augmentation, Normalization).
2. Sclera Segmentation using U-Net architecture.
3. Feature Extraction from the segmented area.
4. Model Training (Random Forest, XGBoost, CNN).
5. Evaluation using metrics like Accuracy, ROC-AUC, and MAE.

## VI. Process Description
The system takes an eye image as input, segments the sclera, extracts dominant color features, and passes them through trained models to provide a diagnostic result.

## VII. Resources and Limitations
- Resources: Python, PyTorch, OpenCV, Scikit-learn, GPU-enabled environment.
- Limitations: Lighting conditions during image capture, variations in skin tone, and the need for high-quality camera images for better accuracy.

## VIII. Testing Technologies used
- Unit testing for processing scripts.
- Model validation using K-fold Cross Validation.
- Performance profiling of real-time inference scripts.

## IX. Conclusion
The proposed system aims to provide a reliable non-invasive alternative to traditional bilirubin testing, significantly improving the accessibility of jaundice screening.

---

# ABSTRACT

Jaundice, a condition resulting from hyperbilirubinemia, is a common clinical manifestation affecting both neonates and adults. The yellowing of the sclera (icterus) is one of the earliest and most visible signs of this condition. Conventional diagnosis relies on Serum Bilirubin (SBR) measurements via invasive blood samples, which poses challenges in neonates and populations with limited access to healthcare. This project explores the development of an "AI-Based Non-Invasive Jaundice Detection" system that utilizes image processing and machine learning to analyze the sclera's color characteristics.

The proposed methodology involves an end-to-end pipeline starting with the precise segmentation of the sclera using a U-Net convolutional neural network. Following segmentation, color features are extracted in multiple color spaces, including RGB, CIELAB, and HSV, to capture the subtle yellowing associated with bilirubin accumulation. These features are then utilized to train a suite of machine learning models, including Random Forest, XGBoost, and a deep Regression neural network, for both jaundice classification and quantitative bilirubin level estimation.

Experimental results demonstrate that the system achieves high accuracy in identifying jaundice and provides bilirubin estimates that correlate strongly with clinical laboratory values. The integration of advanced computer vision techniques with medical diagnostics offers a promising solution for low-cost, non-invasive, and rapid jaundice screening, particularly in remote and underserved regions. The study further discusses the impact of lighting variations and image quality on model performance, providing a foundation for future mobile-based diagnostic applications.
