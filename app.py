import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

st.title("Breast Cancer Classification App")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

models = {
    "Logistic Regression": joblib.load("model/Logistic Regression.pkl"),
    "Decision Tree": joblib.load("model/Decision Tree.pkl"),
    "kNN": joblib.load("model/kNN.pkl"),
    "Naive Bayes": joblib.load("model/Naive Bayes.pkl"),
    "Random Forest": joblib.load("model/Random Forest.pkl"),
    "XGBoost": joblib.load("model/XGBoost.pkl")
}

scaler = joblib.load("model/scaler.pkl")

model_choice = st.selectbox("Select Model", list(models.keys()))

if uploaded_file:
    data = pd.read_csv(uploaded_file)

    st.write("Dataset Preview")
    st.dataframe(data.head())

    model = models[model_choice]

    if model_choice in ["Logistic Regression", "kNN"]:
        data_scaled = scaler.transform(data)
        predictions = model.predict(data_scaled)
    else:
        predictions = model.predict(data)

    st.write("Predictions")
    st.write(predictions)

    if st.checkbox("Show Confusion Matrix"):
        true_labels = st.number_input("Enter True Labels Column Index", step=1)

        y_true = data.iloc[:, true_labels]
        cm = confusion_matrix(y_true, predictions)

        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', ax=ax)
        st.pyplot(fig)

    if st.checkbox("Show Classification Report"):
        st.text(classification_report(y_true, predictions))
