# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load sample churn dataset (replace with actual dataset if needed)
data = pd.read_csv('customer_churn.csv')  # Ensure this file exists

# Preprocessing (adjust based on your actual columns)
X = data.drop('churn', axis=1)
y = data['churn']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the model
with open('churn_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Model trained and saved successfully as churn_model.pkl")
