import streamlit as st
import pandas as pd
import pickle

# Get the directory of the current file
current_dir = os.path.dirname(__file__)
model_path = os.path.join(current_dir, 'churn_model.pkl')

# Load the trained model
with open('churn_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Title
st.title("Employee Churn Prediction App")

# Input fields for user
st.subheader("Fill in the employee details below to predict whether they are likely to leave the company.")

satisfaction_level = st.slider("Satisfaction Level", 0.0, 1.0, 0.5)
last_evaluation = st.slider("Last Evaluation Score", 0.0, 1.0, 0.5)
number_project = st.number_input("Number of Projects", min_value=1, max_value=10, value=3)
average_montly_hours = st.number_input("Average Monthly Hours", min_value=80, max_value=400, value=160)
time_spend_company = st.number_input("Years at Company", min_value=1, max_value=10, value=3)

work_accident = st.selectbox("Had Work Accident?", ["No", "Yes"])
promotion_last_5years = st.selectbox("Promotion in Last 5 Years?", ["No", "Yes"])
department = st.selectbox("Department", ["IT", "RandD", "Accounting", "HR", "Management", "Marketing", "Product_mng", "Sales", "Support", "Technical"])
salary = st.selectbox("Salary Level", ["low", "medium", "high"])

# Preprocessing
work_accident = 1 if work_accident == "Yes" else 0
promotion_last_5years = 1 if promotion_last_5years == "Yes" else 0

# Map department
dept_mapping = {
    "IT": "technical",
    "RandD": "RandD",
    "Accounting": "accounting",
    "HR": "hr",
    "Management": "management",
    "Marketing": "marketing",
    "Product_mng": "product_mng",
    "Sales": "sales",
    "Support": "support",
    "Technical": "technical"
}

department = dept_mapping.get(department, "technical")

# Create DataFrame for prediction
input_data = pd.DataFrame({
    'satisfaction_level': [satisfaction_level],
    'last_evaluation': [last_evaluation],
    'number_project': [number_project],
    'average_montly_hours': [average_montly_hours],
    'time_spend_company': [time_spend_company],
    'Work_accident': [work_accident],
    'promotion_last_5years': [promotion_last_5years],
    'department': [department],
    'salary': [salary]
})

# Predict
if st.button("Predict Churn"):
    prediction = model.predict(input_data)
    if prediction[0] == 1:
        st.error("⚠️ The employee is likely to leave the company.")
    else:
        st.success("✅ The employee is likely to stay in the company.")