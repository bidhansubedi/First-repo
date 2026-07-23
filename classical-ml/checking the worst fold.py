import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

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

# Create the splitter (10 folds)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Get the indices for each fold
fold_number = 0
for train_index, test_index in kfold.split(data, data['hd']):
    fold_number += 1
    
    # Get the patients in this fold's test set
    test_fold = data.iloc[test_index]
    
    print(f"\n{'='*60}")
    print(f"FOLD {fold_number} ANALYSIS")
    print(f"{'='*60}")
    print(f"Number of patients: {len(test_fold)}")
    print(f"Healthy: {(test_fold['hd'] == 0).sum()}")
    print(f"Diseased: {(test_fold['hd'] == 1).sum()}")
    print()
    print(f"Average age: {test_fold['age'].mean():.1f}")
    print(f"Male: {(test_fold['sex'] == 1).sum()}, Female: {(test_fold['sex'] == 0).sum()}")
    print(f"Average cholesterol: {test_fold['chol'].mean():.1f}")
    print(f"Average max heart rate: {test_fold['thalach'].mean():.1f}")
    print(f"Average chest pain type: {test_fold['cp'].mean():.2f}")
    print()