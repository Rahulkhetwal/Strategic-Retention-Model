import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('employee_data.csv')

# Encode categorical variables
le = LabelEncoder()
df['Department'] = le.fit_transform(df['Department'])
df['Salary'] = le.fit_transform(df['Salary'])

# Separate features and target
X = df.drop('left', axis=1)
y = df['left']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Decision Tree Classifier
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train, y_train)
dt_predictions = dt_classifier.predict(X_test)

print("Decision Tree Classifier Results:")
print(classification_report(y_test, dt_predictions))

# Random Forest Classifier
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train, y_train)
rf_predictions = rf_classifier.predict(X_test)

print("\nRandom Forest Classifier Results:")
print(classification_report(y_test, rf_predictions))

# Feature Importance Visualization
plt.figure(figsize=(10, 6))
feature_importance = rf_classifier.feature_importances_
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, X.columns[sorted_idx])
plt.xlabel('Feature Importance')
plt.title('Random Forest Feature Importance')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()

print("\nFeature importance plot saved as 'feature_importance.png'")
