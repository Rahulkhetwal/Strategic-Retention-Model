import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path

# Safe path handling for both local and Streamlit Cloud
current_dir = Path(__file__).parent if '__file__' in globals() else Path.cwd()
model_path = current_dir / 'churn_model.pkl'

# Load the trained model
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Streamlit App Title
st.title("Customer Churn Prediction App")

# Collect user inputs
gender = st.selectbox("Gender", ["Male", "Female"])
SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
Partner = st.selectbox("Has Partner", ["Yes", "No"])
Dependents = st.selectbox("Has Dependents", ["Yes", "No"])
tenure = st.slider("Tenure (months)", 0, 72, 12)
PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, max_value=500.0, value=50.0)
TotalCharges = st.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=500.0)

# Convert inputs into a DataFrame
input_data = pd.DataFrame({
    "gender": [gender],
    "SeniorCitizen": [SeniorCitizen],
    "Partner": [Partner],
    "Dependents": [Dependents],
    "tenure": [tenure],
    "PhoneService": [PhoneService],
    "InternetService": [InternetService],
    "MonthlyCharges": [MonthlyCharges],
    "TotalCharges": [TotalCharges]
})

# Prediction
if st.button("Predict Churn"):
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.error("The customer is likely to churn.")
    else:
        st.success("The customer is not likely to churn.")
