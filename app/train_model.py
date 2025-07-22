import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
df = pd.read_csv('HR_comma_sep.csv')

# One-hot encode categorical features
df_encoded = pd.get_dummies(df, columns=['department', 'salary'], drop_first=True)

# Separate features and target
X = df_encoded.drop('left', axis=1)
y = df_encoded['left']

# Save the feature column names for use in app.py
feature_columns = X.columns.tolist()
joblib.dump(feature_columns, 'feature_columns.pkl')

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'churn_model.pkl')

print("✅ Model training complete.")
print("📁 Model saved as 'churn_model.pkl'")
print("📁 Feature columns saved as 'feature_columns.pkl'")
