import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.stats import chi2

# Load the data (same URL as StatQuest)
url = "https://raw.githubusercontent.com/StatQuest/logistic_regression_demo/master/processed.cleveland.data"
data = pd.read_csv(url, header=None)

# Add column names (exactly like StatQuest's R code)
data.columns = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", 
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "hd"
]

# Convert "?" to NaN (missing values)
data = data.replace("?", np.nan)

# Convert to appropriate types
data['sex'] = data['sex'].astype(int)
data['cp'] = data['cp'].astype(int)
data['fbs'] = data['fbs'].astype(int)
data['restecg'] = data['restecg'].astype(int)
data['exang'] = data['exang'].astype(int)
data['slope'] = data['slope'].astype(int)
data['ca'] = pd.to_numeric(data['ca'], errors='coerce')
data['thal'] = pd.to_numeric(data['thal'], errors='coerce')
data['hd'] = (data['hd'] > 0).astype(int)  # convert to binary: 0=healthy, 1=unhealthy

print(f"Rows before removing missing data: {len(data)}")

# Remove rows with missing ca or thal (like StatQuest did)
data = data.dropna(subset=['ca', 'thal'])

print(f"Rows after removing missing data: {len(data)}")

# Now implement logistic regression from scratch
# Start simple: predict heart disease from sex only (like StatQuest's first model)

X = data[['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
           'restecg', 'thalach', 'exang', 'oldpeak', 'slope']].values  # features: eveeything
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)
y = data['hd'].values     # target: heart disease (0 or 1)

# Add bias term
X = np.column_stack([np.ones(len(X)), X])

# Initialize weights
w = np.zeros(X.shape[1])
learning_rate = 0.001
iterations = 2000

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Cross-entropy loss
def cross_entropy_loss(y, y_pred):
    epsilon = 1e-15
    return -(np.mean(y * np.log(y_pred + epsilon) + (1 - y) * np.log(1 - y_pred + epsilon)))

# Gradient descent
m = len(y)
for i in range(iterations):
    z = X @ w
    y_pred = sigmoid(z)
    
    # Gradient
    dw = (1/m) * X.T @ (y_pred - y)
    
    # Update
    w = w - learning_rate * dw
    
    if i % 100 == 0:
        loss = cross_entropy_loss(y, y_pred)
        print(f"Iteration {i}: loss={loss:.4f}")

print(f"\nFinal weights: bias={w[0]:.4f}, sex coefficient={w[1]:.4f}")

# Make predictions
y_pred = sigmoid(X @ w)

# Calculate accuracy
accuracy = np.mean((y_pred > 0.5) == y)
print(f"Accuracy: {accuracy:.4f}")

# Show predicted probabilities for a few samples
print("\nPredicted probabilities (first 10):")
for i in range(min(10, len(y))):
    print(f"Sex={X[i,1]:.0f}: probability of HD = {y_pred[i]:.4f}, actual = {y[i]}")