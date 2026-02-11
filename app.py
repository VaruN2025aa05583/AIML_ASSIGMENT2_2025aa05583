import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# --- PAGE CONFIG ---
st.set_page_config(page_title="Breast Cancer Classification", layout="wide")

st.title("Breast Cancer Classification App")
st.markdown("ML Assignment 2 | Breast Cancer Classification - 2025aa05583")

# --- 1. UPLOAD DATA ---
uploaded_file = st.file_uploader("Upload CSV file (Test Data)", type=["csv"])


# --- 2. LOAD RESOURCES ---
try:
    models = {
        "Logistic Regression": joblib.load("model/logistic_regression.pkl"),
        "Decision Tree": joblib.load("model/decision_tree.pkl"),
        "kNN": joblib.load("model/knn.pkl"),
        "Naive Bayes": joblib.load("model/naive_bayes.pkl"),
        "Random Forest": joblib.load("model/random_forest.pkl"),
        "XGBoost": joblib.load("model/xgboost.pkl")
    }
    scaler = joblib.load("model/scaler.pkl")
except Exception as e:
    st.error(f"Error loading models/scaler: {e}")
    st.stop()

# --- 3. MODEL SELECTION ---
model_choice = st.selectbox("Select Model", list(models.keys()))
st.write("### Model Insight")
st.info(f"You are currently using **{model_choice}** for prediction.")
# --- 4. MAIN LOGIC ---
if uploaded_file:
    # Read Data
    data = pd.read_csv(uploaded_file)
    st.write("### 1. Dataset Preview")
    st.dataframe(data.head())

    # --- PREPROCESSING ---
    # A. Handle Target Column (if exists)
    if 'diagnosis' in data.columns:
        # Convert M/B to 1/0 for metrics
        y_true = data['diagnosis'].map({'M': 0, 'B': 1}) # Check if your model uses 0=M or 1=M!
        # If your model was trained with 0=Malignant, 1=Benign (sklearn default is 0=M, 1=B)
        # Verify this mapping based on your training! Sklearn load_breast_cancer target: 0=Malignant, 1=Benign
    else:
        y_true = None

    # B. Drop Non-Feature Columns
    cols_to_drop = ['id', 'diagnosis', 'Unnamed: 32']
    X = data.drop(columns=[c for c in cols_to_drop if c in data.columns])

    # C. RENAME COLUMNS to match Scaler's expected names
    # The scaler expects: 'mean radius', 'radius error', 'worst radius'
    # The CSV has: 'radius_mean', 'radius_se', 'radius_worst'
    
    rename_dict = {
        # Mean
        'radius_mean': 'mean radius', 'texture_mean': 'mean texture', 
        'perimeter_mean': 'mean perimeter', 'area_mean': 'mean area', 
        'smoothness_mean': 'mean smoothness', 'compactness_mean': 'mean compactness', 
        'concavity_mean': 'mean concavity', 'concave points_mean': 'mean concave points', 
        'symmetry_mean': 'mean symmetry', 'fractal_dimension_mean': 'mean fractal dimension',
        # Error (SE)
        'radius_se': 'radius error', 'texture_se': 'texture error', 
        'perimeter_se': 'perimeter error', 'area_se': 'area error', 
        'smoothness_se': 'smoothness error', 'compactness_se': 'compactness error', 
        'concavity_se': 'concavity error', 'concave points_se': 'concave points error', 
        'symmetry_se': 'symmetry error', 'fractal_dimension_se': 'fractal dimension error',
        # Worst
        'radius_worst': 'worst radius', 'texture_worst': 'worst texture', 
        'perimeter_worst': 'worst perimeter', 'area_worst': 'worst area', 
        'smoothness_worst': 'worst smoothness', 'compactness_worst': 'worst compactness', 
        'concavity_worst': 'worst concavity', 'concave points_worst': 'worst concave points', 
        'symmetry_worst': 'worst symmetry', 'fractal_dimension_worst': 'worst fractal dimension'
    }
    
    # Apply renaming only if columns match the CSV pattern
    X = X.rename(columns=rename_dict)
    
    # D. Scaling
    model = models[model_choice]
    
    if model_choice in ["Logistic Regression", "kNN"]:
        try:
            # Now X has correct names ('mean radius', etc.)
            X_processed = scaler.transform(X)
        except Exception as e:
            st.error(f"Scaling Error: {e}")
            st.stop()
    else:
        X_processed = X

    # --- 5. PREDICTION ---
    try:
        predictions = model.predict(X_processed)
        
        # Create a readable DataFrame
        results_df = X.copy()
        
        # Add Prediction Column (Map 0/1 back to Text if possible)
        # Assuming 0 = Malignant, 1 = Benign based on standard sklearn dataset
        label_map = {0: 'Malignant', 1: 'Benign'} 
        
        results_df['Prediction'] = [label_map.get(p, p) for p in predictions]
        
        if y_true is not None:
            # Add Actual Column
            results_df['Actual'] = [label_map.get(y, y) for y in y_true]
            
            # Add Status Column (Correct/Incorrect)
            results_df['Status'] = results_df.apply(
                lambda x: '✅ Correct' if x['Actual'] == x['Prediction'] else '❌ Missed', axis=1
            )
            
            # Reorder columns to put results first
            cols = ['Status', 'Actual', 'Prediction'] + [c for c in results_df.columns if c not in ['Status', 'Actual', 'Prediction']]
            results_df = results_df[cols]
            
            # --- COLOR STYLING ---
            def highlight_errors(val):
                color = 'red' if val == '❌ Missed' else 'green'
                return f'color: {color}; font-weight: bold'

            st.write("### 2. Detailed Prediction Results")
            st.dataframe(results_df.style.map(highlight_errors, subset=['Status']))
            
        else:
            # If no ground truth, just show predictions
            st.write("### 2. Predictions")
            st.dataframe(results_df[['Prediction'] + list(X.columns)])
            
    except Exception as e:
        st.error(f"Prediction Error: {e}")
        st.stop()

    # --- 6. EVALUATION ---
    if y_true is not None:
        st.write("### 3. Evaluation Metrics")
        
        # Calculate Metrics
        try:
            acc = accuracy_score(y_true, predictions)
            
            col1, col2 = st.columns(2)
            col1.metric("Accuracy", f"{acc:.2f}")
            
            st.text("Confusion matrix")
            cm = confusion_matrix(y_true, predictions)
            fig, ax = plt.subplots(1,1)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            st.pyplot(fig, use_container_width=False)

            st.text("Classification Report")
            st.text(classification_report(y_true, predictions))
                
        except ValueError as e:
            st.warning(f"Could not calculate metrics. Check target labels mapping. Error: {e}")

else:
    st.info("Please upload a CSV file to proceed.")