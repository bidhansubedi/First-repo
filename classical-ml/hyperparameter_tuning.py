import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load and clean data (same as before)
url = "https://raw.githubusercontent.com/StatQuest/logistic_regression_demo/master/processed.cleveland.data"
data = pd.read_csv(url, header=None)

data.columns = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", 
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "hd"
]

data = data.replace("?", np.nan)
data['ca'] = pd.to_numeric(data['ca'], errors='coerce')
data['thal'] = pd.to_numeric(data['thal'], errors='coerce')
data = data.dropna(subset=['ca', 'thal'])
data['hd'] = (data['hd'] > 0).astype(int)

X = data[['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
           'restecg', 'thalach', 'exang', 'oldpeak', 'slope']].values
y = data['hd'].values

# STEP 1 — Hold out a real test set that we NEVER touch during tuning
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train+validation set: {len(X_trainval)} patients")
print(f"Held-out test set: {len(X_test)} patients (untouched until the very end)")
print()

# STEP 2 — Define the grid of hyperparameters to search
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

total_combinations = 1
for key, values in param_grid.items():
    total_combinations *= len(values)
print(f"Total hyperparameter combinations to try: {total_combinations}")
print(f"With 5-fold CV, total models trained: {total_combinations * 5}")
print()

# STEP 3 — Set up the search
rf = RandomForestClassifier(random_state=42)
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=kfold,
    scoring='recall',  # prioritizing catching actual disease cases
    n_jobs=-1,          # use all CPU cores to speed this up
    verbose=1            # print progress
)

# STEP 4 — Run the search (this trains many models — will take a bit)
print("Running grid search... this may take a minute.")
grid_search.fit(X_trainval, y_trainval)

# STEP 5 — Look at the results
print()
print("=" * 60)
print("BEST HYPERPARAMETERS FOUND")
print("=" * 60)
print(grid_search.best_params_)
print(f"Best cross-validation recall: {grid_search.best_score_:.4f}")
print()

# STEP 6 — Use the ONE final model to evaluate on the untouched test set
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

print("=" * 60)
print("FINAL EVALUATION ON HELD-OUT TEST SET")
print("=" * 60)
print(f"Test accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")

# STEP 7 — Compare to a default (untuned) Random Forest for reference
default_rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
default_rf.fit(X_trainval, y_trainval)
y_pred_default = default_rf.predict(X_test)

print("=" * 60)
print("COMPARISON: Tuned vs Default")
print("=" * 60)
print(f"Default Random Forest test accuracy: {accuracy_score(y_test, y_pred_default):.4f}")
print(f"Tuned Random Forest test accuracy:    {accuracy_score(y_test, y_pred):.4f}")