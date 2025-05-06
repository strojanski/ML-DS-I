import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers

# Generate simple nonlinear data
np.random.seed(0)
data = np.loadtxt("sine.csv", delimiter=",", dtype=float, skiprows=1)

X = data[:, 0]
y = data[:, 1]
X = X.reshape(-1, 1)
X_train  = X[::2]
y_train = y[::2]
X_test = X[1::2]
y_test = y[1::2]

print(X.shape, y.shape)

def rbf_kernel(X1, X2, gamma=2.0):
    # Gaussian (RBF) kernel
    X1_sq = np.sum(X1**2, axis=1)[:, np.newaxis]
    X2_sq = np.sum(X2**2, axis=1)[np.newaxis, :]
    dist_sq = X1_sq + X2_sq - 2 * np.dot(X1, X2.T)
    return np.exp(-gamma * dist_sq)

def polynomial_kernel(X1, X2, degree=7, coef0=2):
    # Polynomial kernel
    return (np.dot(X1, X2.T) + coef0) ** degree

kernel, label, filename = rbf_kernel, 'RBF Kernel', 'kernel-rr-poly.svg'
# kernel, label, filename = rbf_kernel, 'Gaussian (RBF) Kernel', 'kernel-rr-rbf.svg'

# Compute the Gram matrix
K = kernel(X, X)

# Solve for alpha using kernel ridge regression
lmbda = 1e-3  # regularization strength
n = K.shape[0]
alpha = np.linalg.solve(K + lmbda * np.eye(n), y)

# Predict on new data
K_test = kernel(X_test, X)
y_pred = np.dot(K_test, alpha)

# Plot
plt.figure(figsize=(8, 4))
plt.scatter(X, y, color='red', label='Training data')
plt.plot(X_test, y_pred, color='blue', label=label)
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.show()
