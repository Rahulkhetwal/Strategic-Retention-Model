import streamlit as st
import joblib
import pandas as pd

# Load model and feature columns using joblib
model = joblib.load('churn_model.pkl')
feature_cols = joblib.load('feature_columns.pkl')

# Streamlit page configuration
st.set_page_config(page_title="Employee Churn Prediction", page_icon="🧠")

st.markdown("## 🧠 Employee Churn Prediction")
st.markdown("### Fill the employee details below:")

# Input fields
satisfaction_level = st.slider("Satisfaction Level", 0.0, 1.0, step=0.01)
last_evaluation = st.slider("Last Evaluation", 0.0, 1.0, step=0.01)
number_project = st.number_input("Number of Projects", min_value=1, max_value=50, value=3)
average_monthly_hours = st.number_input("Avg Monthly Hours", min_value=0, max_value=500, value=160)
time_spend_company = st.number_input("Time Spent in Company (Years)", min_value=0, max_value=50, value=3)
work_accident = st.selectbox("Work Accident (0 or 1)", options=[0, 1])
promotion_last_5years = st.selectbox("Promotion in Last 5 Years (0 or 1)", options=[0, 1])
department = st.selectbox("Department", options=[
    'sales', 'technical', 'support', 'IT', 'product_mng', 'marketing',
    'RandD', 'accounting', 'hr', 'management'
])
salary = st.selectbox("Salary Level", options=['low', 'medium', 'high'])

# Prepare the input dataframe
input_data = pd.DataFrame([{
    'satisfaction_level': satisfaction_level,
    'last_evaluation': last_evaluation,
    'number_project': number_project,
    'average_montly_hours': average_monthly_hours,
    'time_spend_company': time_spend_company,
    'Work_accident': work_accident,
    'promotion_last_5years': promotion_last_5years,
    'department': department,
    'salary': salary
}])

# One-hot encoding to match training columns
input_data = pd.get_dummies(input_data)
input_data = input_data.reindex(columns=feature_cols, fill_value=0)

# Predict button
if st.button("🔍 Predict"):
    prediction = model.predict(input_data)[0]
    
    if prediction == 1:
        st.error("⚠️ This employee is likely to leave the company.")
    else:
        st.success("✅ This employee is likely to stay with the company.")
