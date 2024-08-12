import streamlit as st
import joblib
import numpy as np

# Define paths to the models
MODEL_PATHS = {
    'Logistic Regression': {
        'GridSearchCV': 'C:/Users/HP/Desktop/Data Science/EXCLER Projects/Bankruptcy Prediction/best_logistic_regression_model.pkl',
        'RandomizedCV': 'C:/Users/HP/Desktop/Data Science/EXCLER Projects/Bankruptcy Prediction/best_logistic_regression_randomcv_model.pkl'
    },
    'SVM': {
        'GridSearchCV': 'C:/Users/HP/Desktop/Data Science/EXCLER Projects/Bankruptcy Prediction/best_svm_model.pkl',
        'RandomizedCV': 'C:/Users/HP/Desktop/Data Science/EXCLER Projects/Bankruptcy Prediction/best_svm_randomcv_model.pkl'
    }
}

# Function to load model
def load_model(model_name, search_type):
    model_path = MODEL_PATHS[model_name][search_type]
    return joblib.load(model_path)

# Streamlit app
st.title("Bankruptcy Prediction")

# Dropdown for model selection
model_name = st.selectbox("Select Model", options=['Logistic Regression', 'SVM'])
search_type = st.selectbox("Select Search Type", options=['GridSearchCV', 'RandomizedCV'])

# Load the selected model
model = load_model(model_name, search_type)

# Helper function to get the risk description
def get_risk_description(value):
    if value == 0:
        return "Low Risk"
    elif value == 0.5:
        return "Medium Risk"
    else:
        return "High Risk"

# Create columns for sliders and their descriptions
col1, col2 = st.columns(2)
with col1:
    industrial_risk = st.slider("Industrial Risk", 0.0, 1.0, 0.0, 0.5)
    st.write(f"**Description**: {get_risk_description(industrial_risk)}")

    management_risk = st.slider("Management Risk", 0.0, 1.0, 0.0, 0.5)
    st.write(f"**Description**: {get_risk_description(management_risk)}")

    financial_flexibility = st.slider("Financial Flexibility", 0.0, 1.0, 0.0, 0.5)
    st.write(f"**Description**: {get_risk_description(financial_flexibility)}")

with col2:
    credibility = st.slider("Credibility", 0.0, 1.0, 0.0, 0.5)
    st.write(f"**Description**: {get_risk_description(credibility)}")

    competitiveness = st.slider("Competitiveness", 0.0, 1.0, 0.0, 0.5)
    st.write(f"**Description**: {get_risk_description(competitiveness)}")

    operating_risk = st.slider("Operating Risk", 0.0, 1.0, 0.0, 0.5)
    st.write(f"**Description**: {get_risk_description(operating_risk)}")

# Hardcode the missing feature value (e.g., 0)
missing_feature = 0.0  # Change to 1.0 if needed

# Convert input values to numpy array including missing feature
input_features = np.array([[industrial_risk, management_risk, financial_flexibility,
                            credibility, competitiveness, operating_risk, missing_feature]])

# Make predictions
if st.button("Check Results"):
    # Make prediction using the loaded model
    prediction = model.predict(input_features)
    prediction_proba = model.predict_proba(input_features)
    
    result = "Bankruptcy" if prediction[0] == 1 else "Non-Bankruptcy"
    probability = prediction_proba[0][1] if prediction[0] == 1 else prediction_proba[0][0]
    probability_percentage = probability * 100
    
    st.write(f"The model predicts: **{result}** with a probability of **{probability_percentage:.2f}%**")
