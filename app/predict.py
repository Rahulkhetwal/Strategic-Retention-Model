import pandas as pd
import pickle

# Load the trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Sample employee data (you can change this to test different inputs)
new_data = pd.DataFrame({
    'satisfaction_level': [0.5],
    'last_evaluation': [0.7],
    'number_project': [3],
    'average_monthly_hours': [160],
    'time_spend_company': [3],
    'Work_accident': [0],
    'churn': [0],
    'promotion_last_5years': [0],
    'department': ['sales'],
    'salary': ['medium']
})

# Perform one-hot encoding just like training
new_data = pd.get_dummies(new_data)

# Load a sample from training to match column order
df = pd.read_csv("employee_data.csv")
df = pd.get_dummies(df)
X = df.drop("churn", axis=1)
expected_cols = X.columns

# Align new data with training columns
new_data = new_data.reindex(columns=expected_cols, fill_value=0)

# Make prediction
prediction = model.predict(new_data)

print("🔮 Prediction:", "Will Leave" if prediction[0] == 1 else "Will Stay")
