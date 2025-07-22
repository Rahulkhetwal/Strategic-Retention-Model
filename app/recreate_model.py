"""Script to recreate the model structure without problematic attributes."""
import os
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import joblib

def recreate_model():
    # Paths
    model_path = os.path.join(os.path.dirname(__file__), 'churn_model.pkl')
    output_path = os.path.join(os.path.dirname(__file__), 'churn_model_compat.pkl')
    
    # Load the original model
    with open(model_path, 'rb') as f:
        original_model = pickle.load(f)
    
    print(f"Original model type: {type(original_model).__name__}")
    
    # Create a new model with the same parameters
    if hasattr(original_model, 'get_params'):
        params = original_model.get_params()
        
        # Create a new model with the same parameters
        if type(original_model).__name__ == 'RandomForestClassifier':
            # Create a simple decision tree without problematic attributes
            class SimpleDecisionTree(DecisionTreeClassifier):
                def __init__(self, **kwargs):
                    # Remove any problematic parameters
                    kwargs.pop('monotonic_cst', None)
                    super().__init__(**kwargs)
            
            # Create a simple random forest
            class SimpleRandomForest(RandomForestClassifier):
                def __init__(self, **kwargs):
                    # Use our simple decision tree as the base estimator
                    kwargs['estimator'] = SimpleDecisionTree()
                    super().__init__(**kwargs)
            
            # Create a new model
            new_model = SimpleRandomForest(**params)
            
            # If the original model has been fitted, copy the estimators
            if hasattr(original_model, 'estimators_'):
                new_model.estimators_ = []
                for est in original_model.estimators_:
                    # Create a simple decision tree for each estimator
                    dt = SimpleDecisionTree(**est.get_params())
                    if hasattr(est, 'tree_'):
                        dt.tree_ = est.tree_
                    if hasattr(est, 'classes_'):
                        dt.classes_ = est.classes_
                    if hasattr(est, 'n_classes_'):
                        dt.n_classes_ = est.n_classes_
                    if hasattr(est, 'n_features_in_'):
                        dt.n_features_in_ = est.n_features_in_
                    if hasattr(est, 'feature_importances_'):
                        dt.feature_importances_ = est.feature_importances_
                    new_model.estimators_.append(dt)
                
                # Copy other important attributes
                for attr in ['classes_', 'n_classes_', 'n_features_in_', 'feature_importances_', 'n_outputs_', 'oob_score_', 'oob_decision_function_']:
                    if hasattr(original_model, attr):
                        setattr(new_model, attr, getattr(original_model, attr))
            
            # Save the new model using joblib which is better for large numpy arrays
            joblib.dump(new_model, output_path)
            print(f"New model saved to: {output_path}")
            return new_model
    
    return None

if __name__ == "__main__":
    print("Recreating model with compatibility fixes...")
    model = recreate_model()
    if model is not None:
        print("Model recreated successfully!")
    else:
        print("Failed to recreate model.")
