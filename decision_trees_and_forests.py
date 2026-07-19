import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

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

# Split into train and test (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")
print()

# ========== DECISION TREE ==========
print("=" * 50)
print("DECISION TREE")
print("=" * 50)

dt = DecisionTreeClassifier(max_depth=None, random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
accuracy_dt = accuracy_score(y_test, y_pred_dt)

print(f"Accuracy: {accuracy_dt:.4f}")
print(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred_dt)}")
print(f"\nClassification Report:\n{classification_report(y_test, y_pred_dt)}")

# Feature importance
print(f"\nFeature Importances:")
feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope']
for name, importance in zip(feature_names, dt.feature_importances_):
    print(f"  {name}: {importance:.4f}")

print()

# ========== RANDOM FOREST ==========
print("=" * 50)
print("RANDOM FOREST (500 trees)")
print("=" * 50)

rf = RandomForestClassifier(n_estimators=500, max_depth=5, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)

print(f"Accuracy: {accuracy_rf:.4f}")
print(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred_rf)}")
print(f"\nClassification Report:\n{classification_report(y_test, y_pred_rf)}")

# Feature importance
print(f"\nFeature Importances:")
for name, importance in zip(feature_names, rf.feature_importances_):
    print(f"  {name}: {importance:.4f}")

print()

# ========== COMPARISON ==========
print("=" * 50)
print("COMPARISON")
print("=" * 50)
print(f"Decision Tree Accuracy: {accuracy_dt:.4f}")
print(f"Random Forest Accuracy: {accuracy_rf:.4f}")
print(f"Improvement: {(accuracy_rf - accuracy_dt):.4f}")