# Strategic Retention Model

A machine learning project to predict employee churn using Decision Trees and Random Forests, featuring interactive controls, visualizations, and flexible CLI options.

---

## 🔍 Project Overview

This project analyzes employee turnover and predicts attrition using Scikit-learn models. It explores the dataset, encodes categorical variables, handles class imbalance, and evaluates model performance. Future enhancements aim to improve flexibility and real-time prediction features.

---

## 🚀 Tasks Breakdown

### ✅ Task 1: Import Libraries
- Imported essential modules from NumPy, Matplotlib, pandas, and Scikit-learn.

### ✅ Task 2: Exploratory Data Analysis
- Loaded and examined the employee dataset.
- Visualized relationships between features and the target variable.

### ✅ Task 3: Encode Categorical Features
- Used dummy encoding for `Department` and `Salary` columns.

### ✅ Task 4: Visualize Class Imbalance
- Used Yellowbrick’s `ClassBalance` visualizer.
- Determined strategy for stratified sampling.

### ✅ Task 5: Create Training and Validation Sets
- Performed 80/20 split using stratified sampling.
- Ensured class balance for model evaluation.

### ✅ Task 6 & 7: Build Decision Tree Classifier
- Created interactive UI with `interact` function.
- Trained a Decision Tree model and visualized the tree.
- Calculated training and validation accuracy.

### ✅ Task 8: Build Random Forest Classifier
- Used `interact` for hyperparameter tuning.
- Built a Random Forest model to reduce variance.
- Compared model performance and visualized a tree.

### ✅ Task 9: Feature Importance and Evaluation
- Extracted and plotted feature importances using `feature_importances_`.
- Evaluated the model with metrics and visual insights.

---

## 🔮 Future Enhancements: Flexible Churn Prediction System

### 💡 Proposed Features:
1. **Interactive Data Input Module**
   - Real-time employee data input and churn predictions.

2. **Enhanced CLI Options**
   - Accept default/custom datasets.
   - Support both manual and batch input modes.

3. **Advanced Prediction & Insights**
   - Risk scoring and detailed reports.
   - Personalized retention strategies.

4. **Seamless Integration**
   - Modular ML pipeline.
   - Input validation and flexible data loading.

---

## 📈 Vision

To evolve this into a robust, user-friendly tool for HR professionals and decision-makers, capable of dynamic employee retention prediction adaptable to any organizational dataset.

---

## 🛠️ Tools & Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Yellowbrick
- Jupyter Notebook
- Git & GitHub

---

## 📌 How to Run

```bash
# Clone the repository
git clone https://github.com/Rahulkhetwal/Strategic-Retention-Model.git

# Navigate into the project
cd Strategic-Retention-Model

# Activate virtual environment (if any)
source venv/bin/activate  # Or use your appropriate path

# Run the notebook or Python script
jupyter notebook  # or python app.py


📬 Contact
Rahul Khetwal
Email: rahulkhetwal00@gmail.com
GitHub: @Rahulkhetwal

⭐ If you like this project, feel free to give it a star on GitHub!