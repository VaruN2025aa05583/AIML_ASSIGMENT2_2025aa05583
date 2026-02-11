# ML Classification Assignment

## Problem Statement
The objective of this project is to build and deploy a machine learning classification pipeline to predict whether a breast tumor is Malignant (Cancerous) or Benign (Non-cancerous) based on cytological features. The solution compares six different classification algorithms and is deployed as an interactive web application using Streamlit.
## Dataset Description
**Dataset:** Breast Cancer Wisconsin (Diagnostic) <br />
**Link to Dataset CSV File:** https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset?resource=download <br />
**Source:** Scikit-learn / UCI Repository <br />
**Features:** 30 features including radius, texture, perimeter, area, smoothness, etc. <br />
**Instances:** 569 samples. <br />
**Classes:** 0 (Malignant), 1 (Benign) <br />
## Models Used & Comparison [cite: 70]

## 3. Models Used & Evaluation Metrics
The following six machine learning models were trained and evaluated on a 20% test split.

| ML Model Name            | Accuracy | AUC Score | Precision | Recall | F1 Score | MCC Score |
|--------------------------|----------|-----------|-----------|--------|----------|-----------|
| **Logistic Regression** | 0.9737   | 0.9974    | 0.9722    | 0.9859 | 0.9790   | 0.9439    |
| **Decision Tree** | 0.9474   | 0.9440    | 0.9577    | 0.9577 | 0.9577   | 0.8880    |
| **k-Nearest Neighbors** | 0.9474   | 0.9820    | 0.9577    | 0.9577 | 0.9577   | 0.8880    |
| **Naive Bayes** | 0.9737   | 0.9984    | 0.9595    | **1.0000** | 0.9793 | 0.9447    |
| **Random Forest** | 0.9649   | 0.9951    | 0.9589    | 0.9859 | 0.9722   | 0.9253    |
| **XGBoost** | *0.9737* | *0.9960* | *0.9722* | *0.9859* | *0.9790* | *0.9439* |

## 4. Observations

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Achieved excellent accuracy (97.37%) and MCC, suggesting the dataset features are largely linearly separable. |
| **Decision Tree** | Showed the lowest AUC (0.9440) among the models, indicating it struggled slightly with probability estimation compared to ensemble methods. |
| **k-NN** | Performed reliably (94.74% Accuracy) but required feature scaling to function correctly. |
| **Naive Bayes** | **Best Model for Medical Safety:** Achieved a perfect Recall score of **1.0000**, meaning it successfully identified 100% of the malignant cases in the test set (zero False Negatives). |
| **Random Forest** | Very stable performance with high AUC (0.9951), balancing precision and recall effectively without much tuning. |
| **XGBoost** | Provided robust results similar to Logistic Regression, effectively handling potential non-linear patterns in the data. |

## 5. Project Structure
```text
ML_Assignment_2/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
└── model/                  # Saved Models & Scaler
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    ├── random_forest.pkl
    ├── xgboost.pkl
    └── scaler.pkl