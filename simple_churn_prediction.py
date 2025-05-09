import csv
import random

def load_data(filename):
    data = []
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    return data

def split_data(data, split_ratio=0.8):
    random.shuffle(data)
    train_size = int(len(data) * split_ratio)
    return data[:train_size], data[train_size:]

def simple_decision_tree_predict(employee):
    # Very basic decision tree logic
    if float(employee['satisfaction_level']) < 0.5:
        return 1  # likely to leave
    if int(employee['number_project']) > 5:
        return 1  # likely to leave
    if float(employee['last_evaluation']) < 0.6:
        return 1  # likely to leave
    return 0  # likely to stay

def evaluate_model(test_data):
    correct_predictions = 0
    for employee in test_data:
        actual = int(employee['left'])
        predicted = simple_decision_tree_predict(employee)
        if actual == predicted:
            correct_predictions += 1
    
    accuracy = correct_predictions / len(test_data)
    return accuracy

def main():
    # Load data
    filename = 'employee_data.csv'
    data = load_data(filename)
    
    # Split data
    train_data, test_data = split_data(data)
    
    # Evaluate model
    accuracy = evaluate_model(test_data)
    
    print(f"Simple Decision Tree Model Accuracy: {accuracy * 100:.2f}%")
    print(f"Total Employees Analyzed: {len(data)}")
    print(f"Training Set Size: {len(train_data)}")
    print(f"Test Set Size: {len(test_data)}")

if __name__ == "__main__":
    main()
