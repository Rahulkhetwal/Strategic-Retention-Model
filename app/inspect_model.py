"""Script to inspect the model's expected features and structure."""
import os
import pickle
import numpy as np

def inspect_model():
    model_path = os.path.join(os.path.dirname(__file__), 'churn_model.pkl')
    
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
            
        print("\n=== Model Information ===")
        print(f"Model type: {type(model).__name__}")
        
        if hasattr(model, 'n_features_in_'):
            print(f"\nExpected number of features: {model.n_features_in_}")
            
        if hasattr(model, 'feature_names_in_'):
            print("\nFeature names:")
            for i, name in enumerate(model.feature_names_in_):
                print(f"{i+1}. {name}")
        
        if hasattr(model, 'estimators_'):
            print(f"\nNumber of estimators: {len(model.estimators_)}")
            if len(model.estimators_) > 0:
                print("\nFirst estimator features:")
                print(f"Number of features: {model.estimators_[0].n_features_in_}")
                if hasattr(model.estimators_[0], 'feature_names_in_'):
                    for i, name in enumerate(model.estimators_[0].feature_names_in_):
                        print(f"{i+1}. {name}")
        
        # Try to get feature importances if available
        if hasattr(model, 'feature_importances_'):
            print("\nFeature importances:")
            if hasattr(model, 'feature_names_in_'):
                for name, importance in sorted(zip(model.feature_names_in_, model.feature_importances_), 
                                           key=lambda x: x[1], reverse=True):
                    print(f"{name}: {importance:.4f}")
            else:
                for i, importance in enumerate(sorted(model.feature_importances_, reverse=True)):
                    print(f"Feature {i}: {importance:.4f}")
                    
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    inspect_model()
