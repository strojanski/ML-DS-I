from cvxopt import matrix, solvers 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

def standardize(X):
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0) + 1e-8  
    
    return (X - X_mean) / X_std, X_mean, X_std

def unstandardize(X, X_mean, X_std):
    return X * X_std + X_mean


class Polynomial:
    def __init__(self, M=2, coef0=1):
        self.degree = M
        self.coef0 = coef0

    def __call__(self, X1, X2):
        X1 = np.atleast_2d(X1)
        X2 = np.atleast_2d(X2)
        result = (np.dot(X1, X2.T) + self.coef0) ** self.degree
        
        if X1.shape[0] == 1 and X2.shape[0] == 1:
            return result[0, 0]  
        if X1.shape[0] == 1 or X2.shape[0] == 1:
            return result.flatten()  
        return result  

class RBF:
    def __init__(self, sigma=1.0):
        self.sigma = sigma

    def __call__(self, X1, X2):
        X1 = np.atleast_2d(X1)
        X2 = np.atleast_2d(X2)
        
        gamma = 1.0 / (2 * self.sigma ** 2)
        
        X1_sq = np.sum(X1**2, axis=1)[:, np.newaxis]
        X2_sq = np.sum(X2**2, axis=1)[np.newaxis, :]
        
        dist_sq = X1_sq + X2_sq - 2 * np.dot(X1, X2.T)
        result = np.exp(-gamma * dist_sq)
        
        
        if X1.shape[0] == 1 and X2.shape[0] == 1:
            return result[0, 0]
        if X1.shape[0] == 1 or X2.shape[0] == 1:
            return result.flatten() 
        return result 


    
class KernelizedRidgeRegression:
    def __init__(self, kernel, lambda_, threshold=1e-6):
        self.kernel = kernel
        self.lambda_ = lambda_
        self.threshold = threshold
        self.C = 1.0 / lambda_
        
    
    def fit(self, X, y):
        X, self.X_mean, self.X_std = standardize(X)
        
        K = self.kernel(X, X)
        self.X = X
        
        n = K.shape[0]
        self.alpha = np.linalg.solve(K + self.lambda_ * np.eye(n), y)
        self.support_vectors = np.where((self.alpha > self.threshold) & (self.alpha < self.C - self.threshold))[0]

        return self

    def predict(self, X_new):
        X_new = (X_new - self.X_mean) / self.X_std
        K_new = self.kernel(X_new, self.X)
        y_pred = K_new @ self.alpha
        return y_pred 
    
class SVR:
    def __init__(self, kernel, lambda_, epsilon, threshold=1e-5):
        self.kernel = kernel
        self.lambda_ = lambda_
        self.epsilon = epsilon
        self.threshold = threshold
        self.C = 1.0 / lambda_
        
    def get_alpha(self):
        return np.column_stack([self.alpha, self.alpha_star])
        
    def get_b(self):
        return self.bias
    
    def fit(self, X, y):
        X, self.X_mean, self.X_std = standardize(X)
        
        self.X = X
        K = self.kernel(X, X)
        n = K.shape[0]
        perm = np.array([i//2 + (i % 2) * n for i in range(2 * n)])
        
        # Quadratic term - 0.5 * [a a*].T @ P @ [a a*] = aKa - 2aKa* + a*Ka* => P = [K -K; -K K]
        P = np.block([
            [K, -K],
            [-K, K]
        ])

        # Linear term = bias
        q = np.concatenate([self.epsilon - y,       # eps - y
                            self.epsilon + y]       # eps + y
                           ).astype(np.double)  

        # Inequality ontraints - between 0 and C (Gx <= h)
        G = np.vstack([-np.eye(2*n), np.eye(2*n)])               # upper bound
        
        h = np.hstack([np.zeros(2*n),           # l = 0
                       np.ones(2*n) * self.C])         # u = C
        
        # # Equality constraints - sum(alpha - alpha*) = 0 (Ax = b)
        A = np.hstack([np.ones((1,n)), -np.ones((1,n))])  # alpha - alpha*
        
        b = np.zeros((1))   # = 0
        
        # Permute so that we get [a1, a1*, a2, a2*, ...]
        P = P[perm][:, perm]
        q = q[perm]
        G = G[perm, :]
        h = h[perm]
        A = A[:, perm]
        
        self.sol = solvers.qp(matrix(P), matrix(q),
                              matrix(G), matrix(h),
                              matrix(A), matrix(b))
        
        x = np.array(self.sol["x"]).flatten()
        self.bias = self.sol["y"][0]
        
        self.alpha = x[::2]
        self.alpha_star = x[1::2]
        
        self.weights = self.alpha - self.alpha_star

        self.support_vectors = self.compute_sv()
        print(self.alpha, self.alpha_star)

        return self

    def compute_sv(self):
        sv = np.where(((self.alpha > self.threshold) & (self.alpha < self.C - self.threshold)) |
                    ((self.alpha_star > self.threshold) & (self.alpha_star < self.C - self.threshold)))[0]

        return sv

    def predict(self, X_new):
        X_new = (X_new - self.X_mean) / self.X_std
        
        K_new = self.kernel(X_new, self.X[self.support_vectors])
        y_pred = K_new @ (self.alpha - self.alpha_star)[self.support_vectors] + float(self.bias)
        print(self.bias)
        
        return y_pred




if __name__ == "__main__":
    data = np.loadtxt("sine.csv", delimiter=",", dtype=float, skiprows=1)
    X = data[:, 0]
    y = data[:, 1]
    
    X = X.reshape(-1, 1)
    y = y.reshape(-1, 1)
    
    
    print(X.shape, y.shape)
    
    kernel = RBF(sigma=0.5)
    kernel = Polynomial(M=11, coef0=1)
    _k = "RBF" if type(kernel) == RBF else "Polynomial"
    
    # ridge = KernelizedRidgeRegression(kernel=kernel, lambda_=1e-3, threshold=1e-4)
    # ridge.fit(X, y)
    # y_pred = ridge.predict(X)
    
    # plt.scatter(X, y, label="Data")
    # plt.scatter(X, y_pred, label=f"Kernel Ridge Regression ({_k})", color="red")
    # plt.scatter(X[ridge.support_vectors], y[ridge.support_vectors], color="green", label="Support Vectors")
    # plt.legend()
    # plt.show()


    svr = SVR(kernel=kernel, lambda_=1e-5, epsilon=.85, threshold=1e-5)
    svr.fit(X, y)
    y_pred = svr.predict(X)
    
    print(len(svr.support_vectors))
    plt.scatter(X, y, label="Data")
    plt.scatter(X, y_pred, label=f"SVR ({_k})", color="red")
    plt.scatter(X[svr.support_vectors], y[svr.support_vectors], color="green", label="Support Vectors")
    plt.legend()
    plt.show()
