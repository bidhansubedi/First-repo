import numpy as np

# Simple dataset — house size vs price
X = np.array([1, 2, 3, 4, 5], dtype=float)  # sizes
y = np.array([1.5, 3.5, 3.0, 4.5, 5.0], dtype=float)  # prices

# Parameters
w = 0.0  # weight (slope)
b = 0.0  # bias (intercept)
learning_rate = 0.01
iterations = 1000
m = len(X)

# Gradient descent loop
for i in range(iterations):
    y_pred = w * X + b           # prediction
    error = y_pred - y           # how wrong are we

    dw = (1/m) * np.dot(error, X)  # gradient for w
    db = (1/m) * np.sum(error)      # gradient for b

    w = w - learning_rate * dw
    b = b - learning_rate * db

    if i % 100 == 0:
        cost = (1/(2*m)) * np.sum(error**2)
        print(f"Iteration {i}: cost={cost:.4f}, w={w:.4f}, b={b:.4f}")

print(f"\nFinal: w={w:.4f}, b={b:.4f}")
for size in [6, 7, 8, 10]:
    print(f"Prediction for size {size}: {w * size + b:.4f}")