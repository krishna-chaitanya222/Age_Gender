

Title: Age and Gender Classification Using CNN

##Introduction
This project implements a deep learning model to classify age and gender from facial images. The goal is to develop a robust system leveraging Convolutional Neural Networks (CNN) for accurate predictions. The system is designed to be scalable and integrates well with real-world applications like demographic analysis, targeted marketing, and enhancing human-computer interaction.

##Dataset Information
UTKFace dataset is a large-scale face dataset with long age span (range from 0 to 116 years old). The dataset consists of over 20,000 face images with annotations of age, gender, and ethnicity. The images cover large variation in pose, facial expression, illumination, occlusion, resolution, etc. This dataset could be used on a variety of tasks, e.g., face detection, age estimation, age progression/regression, landmark localization, etc.

The objective of the project is to detect gender and age using facial images. Convolutional Neural Network is used to classify the images. There are 2 output types namely, gender(M or F) and age.

Download link: https://www.kaggle.com/datasets/jangedoo/utkface-new 

##Methodology
1. Data Collection:
   - Facial images were collected from publicly available datasets with annotated age and gender labels.

2. Preprocessing:
   - Images were resized, normalized, and augmented to improve model performance and prevent overfitting.

3. Model Architecture:
   - A CNN was designed to extract spatial features from images.
   - Batch normalization and dropout layers were incorporated to enhance generalization.

4. Training:
   - The model was trained using categorical cross-entropy loss for gender classification and mean squared error for age regression.
   - Optimizer: Adam with a learning rate scheduler.
   - Training involved multiple epochs with validation to monitor performance.

5. Evaluation:
   - Metrics such as accuracy (for gender) and mean absolute error (for age) were used to assess the model.

6. Deployment:
   - The trained model was saved and can be deployed using a web interface or API for real-time predictions.

##Usage
1. Clone the repository:
   ```
   git clone [repository_url]
   ```
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the notebook:
   Open `age_gender_detection.ipynb` and execute the cells in sequence.

##Future Work
- Enhance dataset size and diversity to improve model robustness.
- Explore advanced architectures like Vision Transformers (ViTs) for better feature extraction.
- Integrate the model into mobile and edge devices for on-the-go predictions.



