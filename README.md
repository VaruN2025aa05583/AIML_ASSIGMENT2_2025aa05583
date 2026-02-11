Problem Statement

The objective of this project is to classify whether a tumor is malignant or benign using machine learning classification models applied to the Breast Cancer Wisconsin (Diagnostic) Dataset.

Dataset Description

The Breast Cancer Wisconsin (Diagnostic) Dataset contains features computed from digitized images of breast mass biopsies.

Number of instances: 569

Number of features: 30

Classification type: Binary (Malignant / Benign)

The features describe characteristics of the cell nuclei present in the image.

Models Used & Performance Comparison
| ML Model Name       | Accuracy | AUC | Precision | Recall | F1 | MCC |
| ------------------- | -------- | --- | --------- | ------ | -- | --- |
| Logistic Regression | XX       | XX  | XX        | XX     | XX | XX  |
| Decision Tree       | XX       | XX  | XX        | XX     | XX | XX  |
| kNN                 | XX       | XX  | XX        | XX     | XX | XX  |
| Naive Bayes         | XX       | XX  | XX        | XX     | XX | XX  |
| Random Forest       | XX       | XX  | XX        | XX     | XX | XX  |
| XGBoost             | XX       | XX  | XX        | XX     | XX | XX  |


(Replace XX using your output)

Model Observations

Logistic Regression
Performed well due to linear separability of features. Scaling improved stability.

Decision Tree
Captured nonlinear relationships but showed slight overfitting tendency.

kNN
Benefited significantly from feature scaling; sensitive to noise.

Naive Bayes
Fast and efficient but assumption of feature independence limits performance.

Random Forest
Improved generalization by reducing overfitting via ensemble averaging.

XGBoost
Delivered strong performance through boosting and sequential error correction.