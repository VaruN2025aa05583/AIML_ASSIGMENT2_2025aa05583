import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef
import joblib
import os

# 1. Load Dataset (Breast Cancer Wisconsin)
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# Check constraints
print(f"Features: {X.shape[1]} (Req >= 12)")
print(f"Instances: {X.shape[0]} (Req >= 500)")

# 2. Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Scaling (Important for KNN and Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler for the app
os.makedirs('model', exist_ok=True)
joblib.dump(scaler, 'model/scaler.pkl')

# 4. Initialize Models [cite: 34-39]
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

# 5. Train, Evaluate, and Save
results = {}

print("\n--- Model Evaluation Metrics ---")
for name, model in models.items():
    # Train
    if name in ["Logistic Regression", "KNN"]:
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        probs = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]
    
    # Calculate Metrics [cite: 40-46]
    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    mcc = matthews_corrcoef(y_test, preds)
    
    results[name] = [acc, auc, prec, rec, f1, mcc]
    
    print(f"\n{name}:")
    print(f"Accuracy: {acc:.4f}, AUC: {auc:.4f}, Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}, F1: {f1:.4f}, MCC: {mcc:.4f}")
    
    # Save Model [cite: 55]
    joblib.dump(model, f'model/{name.replace(" ", "_").lower()}.pkl')

print("\nAll models saved to 'model/' directory.")