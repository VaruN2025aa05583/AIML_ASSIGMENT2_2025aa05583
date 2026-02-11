# ML Classification Assignment

## Problem Statement
To build a machine learning classification pipeline to predict the diagnosis of breast cancer (Malignant/Benign) based on cytological features, and deploy the solution as an interactive web app.

## Dataset Description
**Dataset:** Breast Cancer Wisconsin (Diagnostic)
**Source:** Scikit-learn / UCI Repository
**Features:** 30 features including radius, texture, perimeter, area, smoothness, etc.
**Instances:** 569 samples.
**Classes:** 0 (Malignant), 1 (Benign)

## [cite_start]Models Used & Comparison [cite: 70]

### Model Evaluation Metrics

| ML Model Name             | Accuracy | AUC    | Precision | Recall | F1 Score | MCC    |
|---------------------------|----------|--------|-----------|--------|----------|--------|
| **Logistic Regression** | 0.9737   | 0.9974 | 0.9722    | 0.9859 | 0.9790   | 0.9439 |
| **Decision Tree** | 0.9474   | 0.9440 | 0.9577    | 0.9577 | 0.9577   | 0.8880 |
| **KNN** | 0.9474   | 0.9820 | 0.9577    | 0.9577 | 0.9577   | 0.8880 |
| **Naive Bayes** | 0.9737   | 0.9984 | 0.9595    | 1.0000 | 0.9793   | 0.9447 |
| **Random Forest** | 0.9649   | 0.9951 | 0.9589    | 0.9859 | 0.9722   | 0.9253 |
| **XGBoost** | *Run Locally* | *Run Locally* | *Run Locally* | *Run Locally* | *Run Locally* | *Run Locally* |

*Note: Replace the values above with the actual outputs from your `train_models.py` execution.*

## Observations [cite: 79]

### Model Performance Observations

| ML Model Name             | Observation about model performance |
|---------------------------|-------------------------------------|
| **Logistic Regression** | Achieved high accuracy (97.37%) and MCC, indicating the data is linearly separable. |
| **Decision Tree** | Lower performance (94.74%) compared to ensemble methods, likely due to overfitting. |
| **KNN** | Matched Decision Tree accuracy but had a better AUC (0.9820), showing good ranking ability. |
| **Naive Bayes** | Excellent Recall (1.0000), making it the best model for minimizing false negatives. |
| **Random Forest** | Very stable and high performance (96.49%), balancing precision and recall effectively. |
| **XGBoost** | (Add your observation here once you run it locally or on the cloud). |