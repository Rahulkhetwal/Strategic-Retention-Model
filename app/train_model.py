import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the dataset
data = pd.read_csv('customer_churn.csv')

# Replace 'Exited' with 'Churn' to match your CSV
y = data['Churn']
X = data.drop('Churn', axis=1)

# Encode categorical features
for column in X.columns:
    if X[column].dtype == 'object':
        le = LabelEncoder()
        X[column] = le.fit_transform(X[column])

# Encode the target variable if it's 'Yes'/'No'
if y.dtype == 'object':
    y = y.map({'Yes': 1, 'No': 0})

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'churn_model.pkl')

print("✅ Model trained and saved as churn_model.pkl")
