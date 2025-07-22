import streamlit as st
import pickle
import pandas as pd
import os

# Load model
model_path = os.path.join(os.path.dirname(__file__), 'churn_model.pkl')
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Set page config
st.set_page_config(
    page_title="Strategic Retention Predictor",
    page_icon="📊",
    layout="centered"
)

# Custom CSS for professional look and pointer cursor
st.markdown("""
    <style>
    .main {
        background-color: #f9f9f9;
    }
    div[data-baseweb="select"] > div {
        cursor: pointer;
    }
    button {
        cursor: pointer;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 16px;
        font-size: 16px;
        border-radius: 8px;
    }
    </style>
    """, unsafe_allow_html=True)

# Title
st.title("📉 Strategic Retention Predictor")
st.write("This tool predicts customer churn likelihood based on critical features from telecom industries. Powered by machine learning.")

# Input features
st.header("🔍 Enter Customer Details:")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.selectbox("Senior Citizen", ["Yes", "No"])
    partner = st.selectbox("Has Partner?", ["Yes", "No"])
    dependents = st.selectbox("Has Dependents?", ["Yes", "No"])

with col2:
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    monthly_charges = st.slider("Monthly Charges", 0.0, 200.0, 70.0)
    total_charges = st.slider("Total Charges", 0.0, 10000.0, 1500.0)

# Convert inputs to DataFrame
data = pd.DataFrame({
    'gender': [gender],
    'SeniorCitizen': [1 if senior_citizen == "Yes" else 0],
    'Partner': [partner],
    'Dependents': [dependents],
    'tenure': [tenure],
    'MonthlyCharges': [monthly_charges],
    'TotalCharges': [total_charges]
})

# Prediction
if st.button("📊 Predict Churn"):
    prediction = model.predict(data)[0]
    result = "Yes ❌" if prediction == 1 else "No ✅"
    st.success(f"Churn Prediction: **{result}**")
    if prediction == 1:
        st.warning("⚠️ The customer is likely to churn. Consider retention actions.")
    else:
        st.info("✅ The customer is likely to stay.")

# Footer
st.markdown("---")
st.caption("Built with ❤️ for industry professionals to boost retention and revenue.")
