import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, cross_validate, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the same heart disease dataset
url = "https://raw.githubusercontent.com/StatQuest/logistic_regression_demo/master/processed.cleveland.data"
data = pd.read_csv(url, header=None)

data.columns = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", 
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "hd"
]

# Clean data (same as before)
data = data.replace("?", np.nan)
data['ca'] = pd.to_numeric(data['ca'], errors='coerce')
data['thal'] = pd.to_numeric(data['thal'], errors='coerce')
data = data.dropna(subset=['ca', 'thal'])
data['hd'] = (data['hd'] > 0).astype(int)

# Prepare features and target
X = data[['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
           'restecg', 'thalach', 'exang', 'oldpeak', 'slope']].values
y = data['hd'].values

print("=" * 60)
print("CROSS-VALIDATION WITH 10 FOLDS")
print("=" * 60)
print()

# Create 5-fold cross-validation splitter
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# ========== DECISION TREE ==========
print("DECISION TREE (max_depth=5)")
print("-" * 60)

dt = DecisionTreeClassifier(max_depth=5, random_state=42)

# Run cross-validation with multiple metrics
scoring = {
    'accuracy': 'accuracy',
    'precision': 'precision',
    'recall': 'recall',
    'f1': 'f1'
}

cv_results_dt = cross_validate(dt, X, y, cv=kfold, scoring=scoring)

print(f"Fold accuracies: {cv_results_dt['test_accuracy']}")
print(f"Mean accuracy: {cv_results_dt['test_accuracy'].mean():.4f} (+/- {cv_results_dt['test_accuracy'].std():.4f})")
print(f"Mean precision: {cv_results_dt['test_precision'].mean():.4f}")
print(f"Mean recall: {cv_results_dt['test_recall'].mean():.4f}")
print(f"Mean F1: {cv_results_dt['test_f1'].mean():.4f}")
print()

# ========== RANDOM FOREST ==========
print("RANDOM FOREST (100 trees, max_depth=5)")
print("-" * 60)

rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)

cv_results_rf = cross_validate(rf, X, y, cv=kfold, scoring=scoring)

print(f"Fold accuracies: {cv_results_rf['test_accuracy']}")
print(f"Mean accuracy: {cv_results_rf['test_accuracy'].mean():.4f} (+/- {cv_results_rf['test_accuracy'].std():.4f})")
print(f"Mean precision: {cv_results_rf['test_precision'].mean():.4f}")
print(f"Mean recall: {cv_results_rf['test_recall'].mean():.4f}")
print(f"Mean F1: {cv_results_rf['test_f1'].mean():.4f}")
print()

# ========== COMPARISON ==========
print("=" * 60)
print("COMPARISON")
print("=" * 60)
print(f"Decision Tree mean accuracy: {cv_results_dt['test_accuracy'].mean():.4f}")
print(f"Random Forest mean accuracy: {cv_results_rf['test_accuracy'].mean():.4f}")
print()

# Important insight: check if results are stable
print("STABILITY CHECK (low std = stable across folds):")
print(f"Decision Tree accuracy std: {cv_results_dt['test_accuracy'].std():.4f}")
print(f"Random Forest accuracy std: {cv_results_rf['test_accuracy'].std():.4f}")