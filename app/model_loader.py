"""Safe model loader that handles version incompatibilities."""
import pickle
import numpy as np
from sklearn.tree import DecisionTreeClassifier

class SafeModel:
    def __init__(self, model):
        self.model = model
        self.classes_ = model.classes_
        self.n_classes_ = model.n_classes_
        self.n_features_in_ = model.n_features_in_
        self.feature_importances_ = model.feature_importances_
        
        # Copy the tree structure safely
        if hasattr(model, 'tree_'):
            self.tree_ = SimpleTree(model.tree_)
    
    def predict(self, X):
        if hasattr(X, 'values'):
            X = X.values
        return self.model.predict(X)
    
    def predict_proba(self, X):
        if hasattr(X, 'values'):
            X = X.values
        return self.model.predict_proba(X)

class SimpleTree:
    """A simplified tree structure that doesn't include problematic attributes."""
    def __init__(self, original_tree):
        # Copy only the essential attributes
        self.node_count = original_tree.node_count
        self.capacity = original_tree.capacity
        self.max_depth = original_tree.max_depth
        self.children_left = original_tree.children_left
        self.children_right = original_tree.children_right
        self.feature = original_tree.feature
        self.threshold = original_tree.threshold
        self.value = original_tree.value
        self.n_node_samples = original_tree.n_node_samples
        self.impurity = original_tree.impurity

def load_model_safely():
    """Load the model while avoiding version conflicts."""
    import os
    import warnings
    from sklearn.exceptions import InconsistentVersionWarning
    
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
    
    model_path = os.path.join(os.path.dirname(__file__), 'churn_model.pkl')
    
    try:
        # Load the model with restricted globals to prevent attribute errors
        class RestrictedUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                # Only allow safe classes
                if module == 'numpy.core.multiarray' and name == '_reconstruct':
                    return super().find_class(module, name)
                if module.startswith('numpy') and name in ['ndarray', 'dtype']:
                    return super().find_class(module, name)
                if module == 'sklearn.tree._tree' and name == 'Tree':
                    return SimpleTree
                raise pickle.UnpicklingError(f"global '{module}.{name}' is forbidden")
        
        with open(model_path, 'rb') as f:
            model = RestrictedUnpickler(f).load()
            
        # Wrap the model in our safe container
        return SafeModel(model)
        
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {str(e)}")
