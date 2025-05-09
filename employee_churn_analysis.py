import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

def load_data(filepath):
    """Load employee data from CSV file"""
    try:
        df = pd.read_csv(filepath)
        # Rename 'quit' to 'left' for clarity
        df = df.rename(columns={'quit': 'left'})
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def preprocess_data(df):
    """Preprocess the data for machine learning"""
    # Encode categorical variables
    le = LabelEncoder()
    df['department'] = le.fit_transform(df['department'])
    df['salary'] = le.fit_transform(df['salary'])
    
    # Separate features and target
    X = df.drop('left', axis=1)
    y = df['left']
    
    return X, y

def train_and_evaluate_models(X, y):
    """Train and evaluate Decision Tree and Random Forest models"""
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Decision Tree Classifier
    dt_classifier = DecisionTreeClassifier(random_state=42)
    dt_classifier.fit(X_train_scaled, y_train)
    dt_predictions = dt_classifier.predict(X_test_scaled)
    
    print("Decision Tree Classifier Results:")
    print("Accuracy:", accuracy_score(y_test, dt_predictions))
    print("\nClassification Report:\n", classification_report(y_test, dt_predictions))
    
    # Random Forest Classifier
    rf_classifier = RandomForestClassifier(random_state=42)
    rf_classifier.fit(X_train_scaled, y_train)
    rf_predictions = rf_classifier.predict(X_test_scaled)
    
    print("\nRandom Forest Classifier Results:")
    print("Accuracy:", accuracy_score(y_test, rf_predictions))
    print("\nClassification Report:\n", classification_report(y_test, rf_predictions))
    
    return dt_classifier, rf_classifier, X, X_train, X_test, y_train, y_test, rf_predictions

def main():
    # Load the data
    filepath = 'D:/finalyearmajorpro/Machine-Learning_Deep-learning_Free-Download-381de74cb080305f43ffb710db13f3e6f5ce54e0/A Learner\'s Guide to Model Selection and Tuning/employee_data.csv'
    df = load_data(filepath)
    
    if df is not None:
        # Preprocess the data
        X, y = preprocess_data(df)
        
        # Train and evaluate models
        dt_classifier, rf_classifier, X, X_train, X_test, y_train, y_test, rf_predictions = train_and_evaluate_models(X, y)

        # Confusion Matrix for Random Forest
        cm = confusion_matrix(y_test, rf_predictions)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix - Random Forest')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig('D:/finalyearmajorpro/Machine-Learning_Deep-learning_Free-Download-381de74cb080305f43ffb710db13f3e6f5ce54e0/A Learner\'s Guide to Model Selection and Tuning/confusion_matrix.png')
        plt.close()

        plt.figure(figsize=(10, 6))
        feature_importance = rf_classifier.feature_importances_
        sorted_idx = np.argsort(feature_importance)
        pos = np.arange(sorted_idx.shape[0]) + .5
        plt.barh(pos, feature_importance[sorted_idx], align='center')
        plt.yticks(pos, X.columns[sorted_idx])
        plt.xlabel('Feature Importance')
        plt.title('Random Forest Feature Importance')
        plt.tight_layout()
        plt.savefig('D:/finalyearmajorpro/Machine-Learning_Deep-learning_Free-Download-381de74cb080305f43ffb710db13f3e6f5ce54e0/A Learner\'s Guide to Model Selection and Tuning/feature_importance.png')
        plt.close()

        print("\nVisualizations saved: 'feature_importance.png' and 'confusion_matrix.png'")

if __name__ == '__main__':
    main()

