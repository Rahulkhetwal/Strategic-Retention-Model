import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Load and preprocess data
df = pd.read_csv('employee_data.csv')
df.rename(columns={'quit': 'churn'}, inplace=True)


# Encode categorical columns
le_department = LabelEncoder()
st.write("Columns in dataset:", df.columns.tolist())
df['department'] = le_department.fit_transform(df['department'])

le_salary = LabelEncoder()
df['salary'] = le_salary.fit_transform(df['salary'])


X = df.drop('churn', axis=1)
y = df['churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Streamlit App
st.title("Employee Churn Prediction App")
st.write("Fill in the employee details below to predict whether they are likely to leave the company.")

# User input form
with st.form("prediction_form"):
    satisfaction_level = st.slider("Satisfaction Level", 0.0, 1.0, 0.5)
    last_evaluation = st.slider("Last Evaluation Score", 0.0, 1.0, 0.6)
    number_project = st.number_input("Number of Projects", min_value=1, max_value=10, value=3)
    average_monthly_hours = st.number_input("Average Monthly Hours", min_value=80, max_value=400, value=160)
    time_spend_company = st.number_input("Years at Company", min_value=0, max_value=10, value=3)
    work_accident = st.selectbox("Had Work Accident?", ["No", "Yes"])
    promotion_last_5years = st.selectbox("Promotion in Last 5 Years?", ["No", "Yes"])
    department = st.selectbox("Department", le_department.classes_)
    salary = st.selectbox("Salary Level", le_salary.classes_)

    submit = st.form_submit_button("Predict")

if submit:
    # Encode categorical values
    dept_encoded = le_department.transform([department])[0]
    salary_encoded = le_salary.transform([salary])[0]

    data = np.array([
        satisfaction_level,
        last_evaluation,
        number_project,
        average_monthly_hours,
        time_spend_company,
        1 if work_accident == "Yes" else 0,
        1 if promotion_last_5years == "Yes" else 0,
        dept_encoded,
        salary_encoded
    ]).reshape(1, -1)

    prediction = model.predict(data)
    result = "🚨 This employee is likely to leave." if prediction[0] == 1 else "✅ This employee is likely to stay."
    st.subheader("Prediction Result")
    st.success(result)
    st.write("Available columns:", df.columns.tolist())
    print("Available columns:", df.columns.tolist())
    

