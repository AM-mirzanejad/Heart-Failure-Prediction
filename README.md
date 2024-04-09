# Heart Failure Predictor

<p align="center">
  <img src="https://img.shields.io/badge/Made%20with-Python-0078D4?style=for-the-badge&logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white">
  <img src="https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white">
  <img src="https://img.shields.io/badge/Scikit_Learn-0078D4?style=for-the-badge&logo=scikit-learn&logoColor=white">
</p>

## Overview

The Heart Disease Predictor code is developed to classify whether an individual has heart disease based on specific input parameters. This project utilizes Python along with the scikit-learn and NumPy packages for implementation.



## Model Performance

The initial version of this project implements the SVC (Support Vector Classifier) and Random Forest algorithms to predict heart disease based on patient data. In the first version released on April 9, 2024, the model achieved an accuracy of 87%. This model demonstrates the capability to identify individuals at risk of heart failure with an accuracy of 87%.

In future updates and versions, it is expected that the model's accuracy will continue to improve as more data is collected, and the algorithms are further refined and optimized.

Stay tuned for updates and improvements in upcoming releases!






### Machine Learning Models Used:

- **Support Vector Classifier (SVC)**
- **Random Forest Classifier**
- **K-Nearest Neighbors (KNN) Imputer**
- **MinMaxScaler**
- **GridSearchCV**

### Libraries and Packages Utilized:

- **pandas** (imported as `pd`): For data manipulation and analysis.
- **numpy** (imported as `np`): For numerical operations and array manipulations.
- **sklearn.impute.KNNImputer**: Used for imputing missing values using the K-Nearest Neighbors approach.
- **sklearn.preprocessing.MinMaxScaler**: Utilized for scaling feature values to a specified range.
- **sklearn.model_selection.train_test_split**: Used to split data into training and testing sets.
- **sklearn.svm.SVC**: Support Vector Classifier model from scikit-learn for classification tasks.
- **sklearn.ensemble.RandomForestClassifier**: Random Forest Classifier model for ensemble learning.
- **sklearn.metrics.accuracy_score**: Used to compute the accuracy of the classifier.
- **sklearn.metrics.classification_report**: Generates a detailed classification report.
- **sklearn.model_selection.GridSearchCV**: Used for hyperparameter tuning and model selection.

These packages and models were integrated to build the heart disease prediction functionality within the application.

## Context

Cardiovascular diseases (CVDs) are the number 1 cause of death globally, taking an estimated 17.9 million lives each year, which accounts for 31% of all deaths worldwide. Four out of 5CVD deaths are due to heart attacks and strokes, and one-third of these deaths occur prematurely in people under 70 years of age. Heart failure is a common event caused by CVDs and this dataset contains 11 features that can be used to predict a possible heart disease.

People with cardiovascular disease or who are at high cardiovascular risk (due to the presence of one or more risk factors such as hypertension, diabetes, hyperlipidaemia or already established disease) need early detection and management wherein a machine learning model can be of great help.

## Heart Disease Dataset

| Feature         | Description                                                                                         |
|-----------------|-----------------------------------------------------------------------------------------------------|
| Age             | Age of the patient (in years)                                                                       |
| Sex             | Sex of the patient (M: Male, F: Female)                                                             |
| Chest Pain Type | Chest pain type (TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic) |
| Resting BP      | Resting blood pressure (mm Hg)                                                                      |
| Cholesterol     | Serum cholesterol level (mm/dl)                                                                     |
| Fasting Blood Sugar | Fasting blood sugar level (1: if FastingBS > 120 mg/dl, 0: otherwise)                               |
| Resting ECG     | Resting electrocardiogram results (Normal: Normal, ST: ST-T wave abnormality, LVH: Left ventricular hypertrophy) |
| Max Heart Rate  | Maximum heart rate achieved (numeric value between 60 and 202)                                       |
| Exercise Angina | Exercise-induced angina (Y: Yes, N: No)                                                             |
| ST Depression   | ST depression induced by exercise (numeric value measured in depression)                             |
| ST Slope        | The slope of the peak exercise ST segment (Up: upsloping, Flat: flat, Down: downsloping)             |
| Heart Disease   | Output class (1: heart disease, 0: Normal)                                                          |

## Installation

Clone the repository using `git`:

```bash
git clone <git url>
cd <directory_name>
pip install -r requirements.txt
