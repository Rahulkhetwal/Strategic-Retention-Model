"""Script to inspect the model structure."""
import os
import pickle

def inspect_model():
    model_path = os.path.join(os.path.dirname(__file__), 'churn_model.pkl')
    
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
            
        print("\n=== Model Information ===")
        print(f"Model type: {type(model).__name__}")
        print("\nModel attributes:")
        for attr in dir(model):
            if not attr.startswith('_') and not callable(getattr(model, attr)):
                print(f"- {attr}: {type(getattr(model, attr)).__name__}")
                
        if hasattr(model, 'tree_'):
            print("\nTree attributes:")
            tree = model.tree_
            for attr in dir(tree):
                if not attr.startswith('_'):
                    print(f"- {attr}")
                    
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    inspect_model()
