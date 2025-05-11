from cvxopt import matrix, solvers 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

def standardize(X):
    X_mean = np.mean(X)
    X_std = np.std(X)
    
    return (X - X_mean) / X_std, X_mean, X_std

def unstandardize(X, X_mean, X_std):
    return X * X_std + X_mean

def polynomial_kernel(x1, x2, coef=1, M=2):
    return (np.dot(x1, x2.T) + coef) ** M

def rbf_kernel(X1, X2, gamma=1.0):
    X1_sq = np.sum(X1**2, axis=1)[:, np.newaxis]
    X2_sq = np.sum(X2**2, axis=1)[np.newaxis, :]
    dist_sq = X1_sq + X2_sq - 2 * np.dot(X1, X2.T)
    return np.exp(-gamma * dist_sq)

def gram_mtx(X, kernel, M=11, gamma=1):
    return kernel(X, X, M=M) if kernel == polynomial_kernel else kernel(X, X, gamma=gamma)

def ridge_regression(X, y, lmbda, kernel="polynomial", M=11):
    kernel = polynomial_kernel if kernel == "polynomial" else rbf_kernel
    K = gram_mtx(X, kernel, M)
    n = K.shape[0]
    

    alpha = np.linalg.solve(K + lmbda * np.eye(n), y)
    
    return alpha, kernel

   
def svr(X, y, lmbda, kernel="polynomial", eps=1e-3, M=11, gamma=0.4):
    # 0.5 [alpha, alpha*].T @ P @ [alpha, alpha*] + q.T @ [alpha, alpha*]
    # K = similarity(X, X) in kernel space
    
    l = X.shape[0]
    C = 1.0 / lmbda
    
    perm = np.array([i//2 + (i % 2) * l for i in range(2 * l)])

    K = polynomial_kernel if kernel=="polynomial" else rbf_kernel
    K = gram_mtx(X, K, M, gamma)
    K += 1e-8 * np.eye(K.shape[0])

    # Quadratic term - 0.5 * [a a*].T @ P @ [a a*] = aKa - 2aKa* + a*Ka* => P = [K -K; -K K]
    P = np.vstack([
        np.hstack([ K, -K ]),
        np.hstack([ -K,  K ])
    ])

    # Linear term = bias
    q = np.vstack([eps * np.ones((l,1)) - y,    # eps - y
                   eps * np.ones((l,1)) + y])   # eps + y

    # Inequality ontraints - between 0 and C (Gx <= h)
    G = np.vstack([-np.eye(2*l),                 # lower bound
                   np.eye(2*l)])               # upper bound
    
    h = np.vstack([np.zeros((2*l,1)),           # l = 0
                   C*np.ones((2*l,1))])         # u = C
    
    # Equality constraints - sum(alpha - alpha*) = 0 (Ax = b)
    A = np.hstack([np.ones((1,l)),          # alpha 
                   -np.ones((1,l))])        # - alpha*
    b = np.zeros((1,1))                     # = 0
    
    # Permute so that we get [a1, a1*, a2, a2*, ...]
    P = P[perm][:, perm]
    q = q[perm]
    G = G[:, perm]
    A = A[:, perm]

    sol = solvers.qp(matrix(P), matrix(q),
                     matrix(G), matrix(h),
                     matrix(A), matrix(b))
    
    z = np.array(sol['x']).flatten()
    
    alpha     =  z[::2]
    alpha_star = z[1::2]

    weights = alpha - alpha_star

    def compute_bias(K, y, alpha, alpha_star, C, thresh=1e-4):
        weights = alpha - alpha_star
        sv = np.where(((alpha > thresh) & (alpha < C - thresh)) |
                    ((alpha_star > thresh) & (alpha_star < C - thresh)))[0]
        
        if len(sv) == 0:
            sv = np.where(np.abs(weights) > 1e-4)[0]  # fallback
        
        b_vals = []
        for i in sv:
            sum_k = np.sum(weights * K[:, i])
            b_vals.append(y[i] - sum_k)
        return float(np.mean(b_vals)), sv

    b_val, support_vectors = compute_bias(K, y, alpha, alpha_star, C, thresh=1e-4)
    print(len(support_vectors), "support vectors")

    def predict(X_new, M=11):
        K_new = polynomial_kernel if kernel=="polynomial" else rbf_kernel
        K_new = K_new(X_new, X[support_vectors], M) if kernel == "polynomial" else K_new(X_new, X[support_vectors], gamma=gamma)  
        
        return K_new @ weights[support_vectors] + b_val

    return alpha, alpha_star, b_val, predict, support_vectors


def krr_train_mse(X, y, kernel_type="rbf", lambdas=[1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1], gammas=[0.1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], Ms=[2, 3, 5, 7, 9, 11]):

    best_score = float("inf")
    best_params = {}

    for lmbda in lambdas:
        if kernel_type == "rbf":
            for gamma in gammas:
                K = rbf_kernel(X, X, gamma)
                alpha = np.linalg.solve(K + lmbda * np.eye(len(X)), y)
                y_pred = K @ alpha
                mse = mean_squared_error(y, y_pred)

                if mse < best_score:
                    best_score = mse
                    best_params = {"lmbda": lmbda, "gamma": gamma}
                    print("rbf", best_params, "MSE:", best_score)
                    plt.scatter(X, y_pred)
                    plt.scatter(X, y)
                    plt.show()

        else:  # polynomial
            for M in Ms:
                K = polynomial_kernel(X, X, M)
                alpha = np.linalg.solve(K + lmbda * np.eye(len(X)), y)
                y_pred = K @ alpha
                mse = mean_squared_error(y, y_pred)

                if mse < best_score:
                    best_score = mse
                    best_params = {"lmbda": lmbda, "M": M}
                    print("poly", best_params, "MSE:", best_score)
                    
                    plt.scatter(X, y_pred)
                    plt.scatter(X, y)
                    plt.show()

    return best_params, best_score




if __name__ == "__main__":
    data = np.loadtxt("sine.csv", delimiter=",", dtype=float, skiprows=1)
    X = data[:, 0]
    y = data[:, 1]
    

    X = X.reshape(-1, 1)
    y = y.reshape(-1, 1)
    
    
    print(X.shape, y.shape)
    # y = y.reshape(-1, 1)
    
    kernel_ = "polynomial"
    # best_params, best_score = krr_train_mse(X, y, kernel_type=kernel_)
    # print(best_params, best_score)

    
    lmbda = 1e-7 if kernel_ == "polynomial" else 1e-4
    M = 7
    
    X, X_mean, X_std = standardize(X)
    y, y_mean, y_std = standardize(y)
    
    
    alpha, kernel = ridge_regression(X, y.reshape(-1, 1), lmbda, kernel=kernel_, M=M)
    support_vectors = np.where((alpha > 1e-6) & (alpha < 1 / lmbda - 1e-6))[0]
    
    K = kernel(X, X, M=M) if kernel_ == "polynomial" else kernel(X, X, gamma=1)
    y_plot = K @ alpha
    
    X = unstandardize(X, X_mean, X_std)
    y_plot = unstandardize(y_plot, y_mean, y_std)
    y = unstandardize(y, y_mean, y_std)
    

    # plt.scatter(X, y, label="Data")
    # plt.scatter(X, y_plot, label=f"Kernel Ridge Regression ({kernel_})", color="red")
    # plt.scatter(X[support_vectors], y[support_vectors], color="green", label="Support Vectors")
    # plt.legend()
    # plt.show()



    data = np.loadtxt("sine.csv", delimiter=",", dtype=float, skiprows=1)
    X = data[:, 0]
    y = data[:, 1]
    

    X = X.reshape(-1, 1)
    y = y.reshape(-1, 1)


    lmbda_rbf = 1e-3
    lmbda = 1e-7
    M = 5
    X, X_mean, X_std = standardize(X)
    y, y_mean, y_std = standardize(y)
    
    alpha, alpha_star, b, predict, sv = svr(X, y, lmbda=lmbda, kernel=kernel_, eps=.001, M=M, gamma=1)

    
    y_plot = predict(X, M=M)

    X = unstandardize(X, X_mean, X_std)
    y_plot = unstandardize(y_plot, y_mean, y_std)
    y = unstandardize(y, y_mean, y_std)

    plt.scatter(X, y, label="Data")
    plt.scatter(X, y_plot, color="red", label=f"SVR Prediction ({kernel_})")
    plt.scatter(X[sv], y[sv], color="green", label="Support Vectors")
    plt.legend()
    plt.show()
