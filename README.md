# Heart Failure Predictor

<p align="center">
  <img src="https://img.shields.io/badge/Made%20with-Python-0078D4?style=for-the-badge&logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white">
  <img src="https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white">
  <img src="https://img.shields.io/badge/Scikit_Learn-0078D4?style=for-the-badge&logo=scikit-learn&logoColor=white">
</p>

---

## Overview

The Heart Disease Predictor is a Python project developed to classify whether an individual has heart disease based on specific input parameters. It utilizes the scikit-learn and NumPy libraries for implementation.

---

## Model Performance

The initial version of this project implements the SVC (Support Vector Classifier) and Random Forest algorithms to predict heart disease based on patient data. The model, released on April 9, 2024, achieved an accuracy of 87% in identifying individuals at risk of heart failure.

Future updates are expected to enhance the model's accuracy through further data collection and algorithm optimization.

Stay tuned for updates and improvements in upcoming releases!

---

## Jupyter Notebook

For a detailed demonstration and usage of the Heart Failure Predictor, refer to the Jupyter Notebook:

[![Jupyter Notebook](https://img.shields.io/badge/Open%20in-Jupyter%20Notebook-orange?style=for-the-badge&logo=jupyter)](https://github.com/AM-mirzanejad/Heart-Failure-Prediction/blob/main/Heart-Prediction.ipynb)

---




### Machine Learning Models Used:

- **Support Vector Classifier (SVC):**
  - Description: SVC is a supervised learning model used for classification tasks. It works by finding the hyperplane that best separates different classes in the feature space.
  - Implementation: Utilized the `sklearn.svm.SVC` class from scikit-learn.

- **Random Forest Classifier:**
  - Description: Random Forest is an ensemble learning method that constructs multiple decision trees during training and outputs the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees.
  - Implementation: Used the `sklearn.ensemble.RandomForestClassifier` class from scikit-learn.

- **K-Nearest Neighbors (KNN) Imputer:**
  - Description: KNN Imputer is used for imputing missing values by using the K-Nearest Neighbors approach, where missing values are imputed based on the values of neighboring data points.
  - Implementation: Utilized the `sklearn.impute.KNNImputer` class from scikit-learn.

- **MinMaxScaler:**
  - Description: MinMaxScaler is used for scaling feature values to a specified range, usually between 0 and 1.
  - Implementation: Used the `sklearn.preprocessing.MinMaxScaler` class from scikit-learn.

- **GridSearchCV:**
  - Description: GridSearchCV is used for hyperparameter tuning and model selection by exhaustively searching through a specified parameter grid and cross-validating the results.
  - Implementation: Utilized the `sklearn.model_selection.GridSearchCV` class from scikit-learn.

---

### Libraries and Packages Utilized:

- **pandas** (imported as `pd`):
  - Description: pandas is a powerful library for data manipulation and analysis, providing data structures and operations for manipulating numerical tables and time series.
  - Icon: <img src="https://img.icons8.com/color/48/000000/pandas.png" width="50" height="50"/>

- **numpy** (imported as `np`):
  - Description: numpy is a fundamental package for scientific computing with Python, providing support for large, multi-dimensional arrays and matrices, along with a collection of mathematical functions to operate on these arrays.
  - Icon: <img src="https://img.icons8.com/color/48/000000/numpy.png" width="50" height="50"/>

- **scikit-learn** (imported as `sklearn`):
  - Description: scikit-learn is a popular machine learning library in Python that provides simple and efficient tools for data mining and data analysis, including a wide variety of machine learning algorithms and utilities for model selection, evaluation, and preprocessing.
  - Icon: <img src="https://icon.icepanel.io/Technology/svg/scikit-learn.svg" width="50" height="50"/>










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
