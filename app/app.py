import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("churn_model.pkl")

# App title and description
st.set_page_config(page_title="Employee Churn Predictor", layout="centered")

st.title("🔍 Employee Churn Prediction App")
st.markdown("""
Welcome to the **Employee Churn Predictor**.  
Fill in the employee details below to predict whether the employee is likely to leave the company.
""")

# Sidebar for dataset column info (optional)
with st.expander("See Dataset Columns"):
    st.write([
        "1. satisfaction_level",
        "2. last_evaluation",
        "3. number_project",
        "4. average_monthly_hours",
        "5. time_spend_company",
        "6. Work_accident",
        "7. promotion_last_5years",
        "8. department",
        "9. salary"
    ])

# Input form
with st.form("prediction_form"):
    st.subheader("📋 Enter Employee Details")

    col1, col2 = st.columns(2)

    with col1:
        satisfaction_level = st.slider("Satisfaction Level", 0.0, 1.0, 0.5)
        number_project = st.number_input("Number of Projects", min_value=1, max_value=10, value=3)
        time_spend_company = st.number_input("Years at Company", min_value=0, max_value=20, value=3)
        work_accident = st.selectbox("Had Work Accident?", ["No", "Yes"])

    with col2:
        last_evaluation = st.slider("Last Evaluation Score", 0.0, 1.0, 0.7)
        average_monthly_hours = st.number_input("Average Monthly Hours", min_value=50, max_value=400, value=160)
        promotion_last_5years = st.selectbox("Promotion in Last 5 Years?", ["No", "Yes"])
        department = st.selectbox("Department", [
            "IT", "RandD", "Accounting", "HR", "Management", 
            "Marketing", "Product Management", "Sales", "Support", "Technical"
        ])
        salary = st.selectbox("Salary Level", ["low", "medium", "high"])

    submitted = st.form_submit_button("Predict")

# Prediction logic
if submitted:
    # Preprocess input
    data = pd.DataFrame({
        "satisfaction_level": [satisfaction_level],
        "last_evaluation": [last_evaluation],
        "number_project": [number_project],
        "average_montly_hours": [average_monthly_hours],
        "time_spend_company": [time_spend_company],
        "Work_accident": [1 if work_accident == "Yes" else 0],
        "promotion_last_5years": [1 if promotion_last_5years == "Yes" else 0],
        "department": [department],
        "salary": [salary]
    })

    # Predict
    prediction = model.predict(data)[0]

    # Display result
    if prediction == 1:
        st.error("⚠️ The employee is likely to leave the company.")
    else:
        st.success("✅ The employee is likely to stay with the company.")
