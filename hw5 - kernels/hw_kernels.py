from cvxopt import matrix, solvers 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_validate, GridSearchCV, KFold

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
        
    def get_params(self, deep=True):
        return {"kernel": self.kernel, "lambda_": self.lambda_, "threshold": self.threshold}
    
    def set_params(self, kernel, lambda_, threshold):
        self.kernel = kernel
        self.lambda_ = lambda_
        self.threshold = threshold
        self.C = 1.0 / lambda_
        return self
    
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
        
    def get_params(self, deep=True):
        return {"kernel": self.kernel, "lambda_": self.lambda_, "epsilon": self.epsilon, "threshold": self.threshold}
    
    def set_params(self, kernel, lambda_, epsilon, threshold):
        self.kernel = kernel
        self.lambda_ = lambda_
        self.epsilon = epsilon
        self.threshold = threshold
        self.C = 1.0 / lambda_
        return self
    
    def get_alpha(self):
        return np.column_stack([self.alpha, self.alpha_star])
        
    def get_b(self):
        return self.bias
    
    def fit(self, X, y):
        X, self.X_mean, self.X_std = standardize(X)
        
        self.X = X
        self.y = y
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
        perm_ = np.array([i//2 + (i % 2) * n for i in range(4 * n)])
        
        # # Equality constraints - sum(alpha - alpha*) = 0 (Ax = b)
        A = np.hstack([np.ones((1,n)), -np.ones((1,n))])  # alpha - alpha*
        b = np.zeros((1))   # = 0
        # Permute so that we get [a1, a1*, a2, a2*, ...]
        P = P[perm][:, perm]
        q = q[perm]
        G = G[perm_, :]
        h = h[perm_]
        A = A[:, perm]
        
        self.sol = solvers.qp(matrix(P), matrix(q),
                              matrix(G), matrix(h),
                              matrix(A), matrix(b), options={"show_progress": False})
        
        x = np.array(self.sol["x"]).flatten()
        self.bias = self.sol["y"][0]
        
        self.alpha = x[::2]
        self.alpha_star = x[1::2]
        
        self.weights = self.alpha - self.alpha_star

        self.support_vectors = self.compute_sv()

        return self

    def compute_sv(self):
        sv = np.where(((self.alpha > self.threshold) & (self.alpha < self.C - self.threshold)) |
                    ((self.alpha_star > self.threshold) & (self.alpha_star < self.C - self.threshold)))[0]
        if len(sv) == 0:
            sv = np.where(np.abs(self.weights) > 1e-4)[0]
        if len(sv) == 0:
            print("No support vectors found. Using all points.")
            sv = np.arange(len(self.alpha))
        return sv

    def predict(self, X_new):
        X_new = (X_new - self.X_mean) / self.X_std
        K_new = self.kernel(X_new, self.X[self.support_vectors])
        
        if K_new.ndim == 1:
            K_new = K_new.reshape(-1, 1)
            
        y_pred = K_new @ self.weights[self.support_vectors] + float(self.bias)
        
        return y_pred




if __name__ == "__main__":
    data = np.loadtxt("sine.csv", delimiter=",", dtype=float, skiprows=1)
    X = data[:, 0]
    y = data[:, 1]
    
    X = X.reshape(-1, 1)
    y = y.reshape(-1, 1)
    
    ###################### P1 ######################
    
    print(X.shape, y.shape)
    
    kernel = Polynomial(M=11, coef0=1)
    kernel = RBF(sigma=0.5)
    _k = "RBF" if type(kernel) == RBF else "Polynomial"
    
    ridge = KernelizedRidgeRegression(kernel=kernel, lambda_=1e-3, threshold=1e-4)
    ridge.fit(X, y)
    y_pred = ridge.predict(X)
    
    plt.scatter(X, y, label="Data")
    plt.scatter(X, y_pred, label=f"Kernel Ridge Regression ({_k})", color="red")
    plt.scatter(X[ridge.support_vectors], y[ridge.support_vectors], color="green", label="Support Vectors")
    plt.legend()
    plt.show()


    svr = SVR(kernel=kernel, lambda_=1e-5, epsilon=.6, threshold=1e-5)
    svr.fit(X, y)
    y_pred = svr.predict(X)
    
    print(len(svr.support_vectors))
    plt.scatter(X, y, label="Data")
    plt.scatter(X, y_pred, label=f"SVR ({_k})", color="red")
    plt.scatter(X[svr.support_vectors], y[svr.support_vectors], color="green", label="Support Vectors")
    plt.legend()
    plt.show()
    ################################################
    
    ###################### P2 ######################

    cols = np.loadtxt("housing2r.csv", delimiter=",", dtype=str,  max_rows=1)
    data = np.loadtxt("housing2r.csv", delimiter=",", dtype=float, skiprows=1)
    X = data[:, :-1]
    y = data[:, -1]
    X = X.reshape(-1, 5)
    y = y.reshape(-1, 1)

    print(cols)
    sigmas = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10]
    lambdas = [1e-3, 1e-2, 1e-1, 1, 10]
    
    Ms = [i for i in range(1, 11)]
    # Cross validation: 1 CV with testing different M and sigma, lambda=1, and 1 nested with best lambda
    
    mse_lr_poly_nocv, se_lr_poly_nocv = [], []
    mse_svr_poly_nocv, se_svr_poly_nocv = [], []
    n_support_vectors_nocv, n_support_vectors_inner = [],  []

    inner_mse_lr_poly = []
    inner_mse_svr_poly = []
    inner_lambda_lr_poly = []
    inner_lambda_svr_poly = []
    k = 10
    # for M in Ms:
    #     kfold = KFold(n_splits=k, shuffle=True, random_state=42)
        
    #     # Normal cross validation for M in [1, 10] and lambda=1
    #     print("M:", M)
    #     cv_lr = cross_validate(KernelizedRidgeRegression(kernel=Polynomial(M=M, coef0=1), lambda_=1, threshold=1e-5), X, y, scoring='neg_mean_squared_error', return_estimator=True, n_jobs=-1, cv=kfold)
    #     cv_svr = cross_validate(SVR(kernel=Polynomial(M=M, coef0=1), lambda_=1, epsilon=.5, threshold=1e-5), X, y, scoring='neg_mean_squared_error', return_estimator=True, n_jobs=-1, cv=kfold)
        
    #     n_support_vectors_nocv.append(sum([len(est.support_vectors) for est in cv_svr['estimator']]) / k)
        
    #     mse_lr_poly_nocv.append(-cv_lr['test_score'].mean())
    #     mse_svr_poly_nocv.append(-cv_svr['test_score'].mean())
        
    #     se_lr_poly_nocv.append(cv_lr['test_score'].std() / np.sqrt(k))
    #     se_svr_poly_nocv.append(cv_svr['test_score'].std() / np.sqrt(k))
        
    #     print("MSE LR Poly:", mse_lr_poly_nocv[-1], "SE LR Poly:", se_lr_poly_nocv[-1])
    #     print("MSE SVR Poly:", mse_svr_poly_nocv[-1], "SE SVR Poly:", se_svr_poly_nocv[-1])
    #     print("N Support Vectors:", n_support_vectors_nocv[-1])
    #     print("=====================================")

    #     # Nested cross validation for best lambda
    #     mse_lr_poly = []
    #     mse_svr_poly = []
    #     lambda_lr_poly = []
    #     lambda_svr_poly = []
        
        
    #     for i, (train_ix, test_ix) in enumerate(kfold.split(X)):
    #         best_mse_lr = float("inf")
    #         best_mse_svr = float("inf")
    #         min_sv_count = float("inf")
    #         X_train, X_test = X[train_ix], X[test_ix]
    #         y_train, y_test = y[train_ix], y[test_ix]
            
    #         best_lambda_lr = None
    #         best_lambda_svr = None
            
    #         parameters_lr = {"kernel": [Polynomial(M=M, coef0=1)], "lambda_": lambdas, "threshold": [1e-5]}
    #         parameters_svr = {"kernel": [Polynomial(M=M, coef0=1)], "lambda_": lambdas, "epsilon":[0.5], "threshold": [1e-5]}
            
    #         ### INNER CV ###
            
    #         gridcv_lr = GridSearchCV(KernelizedRidgeRegression(Polynomial(M=M, coef0=1), lambda_=1), parameters_lr, cv=10, scoring='neg_mean_squared_error', n_jobs=-1)
    #         gridcv_svr = GridSearchCV(SVR(Polynomial(M=M, coef0=1), lambda_=1, epsilon=0.5), parameters_svr, cv=10, scoring='neg_mean_squared_error', n_jobs=-1)
            
    #         gridcv_lr.fit(X_train, y_train)
    #         gridcv_svr.fit(X_train, y_train)
            
    #         print(f"Best MSE LR: {-gridcv_lr.best_score_}, Best MSE SVR: {-gridcv_svr.best_score_}")
    #         print(f"Best Lambda LR: {gridcv_lr.best_params_['lambda_']}, Best Lambda SVR: {gridcv_svr.best_params_['lambda_']}")
               
    #         if gridcv_lr.best_score_ < best_mse_lr:
    #             best_mse_lr = gridcv_lr.best_score_
    #             best_lambda_lr = gridcv_lr.best_params_['lambda_']
    #             best_model_lr = gridcv_lr.best_estimator_
                
    #         if gridcv_svr.best_score_ < best_mse_svr:
    #             best_mse_svr = gridcv_svr.best_score_
    #             best_lambda_svr = gridcv_svr.best_params_['lambda_']
    #             best_model_svr = gridcv_svr.best_estimator_
                
    #         ### OUTER CV ###
    #         y_pred_lr = best_model_lr.predict(X_test)
    #         y_pred_svr = best_model_svr.predict(X_test)
    #         n_support_vectors_inner.append(len(best_model_svr.support_vectors))
    #         mse_lr = mean_squared_error(y_test, y_pred_lr)  
    #         mse_svr = mean_squared_error(y_test, y_pred_svr)
    #         mse_lr_poly.append(mse_lr)
    #         mse_svr_poly.append(mse_svr)
    #         lambda_lr_poly.append(best_lambda_lr)
    #         lambda_svr_poly.append(best_lambda_svr)
            
    #         print(len(mse_svr_poly), "MSE LR:", mse_lr, "MSE SVR:", mse_svr)
    #         print(len(lambda_lr_poly))
            
    #         print(f"Fold {i+1}: Best Lambda LR: {best_lambda_lr} -- MSE LR (with best model): {mse_lr}, Best Lambda SVR: {best_lambda_svr} -- MSE SVR (with best model): {mse_svr}")
    #         print("=====================================")
        
    #     inner_lambda_lr_poly.extend(lambda_lr_poly)
    #     inner_lambda_svr_poly.extend(lambda_svr_poly)
    #     inner_mse_lr_poly.extend(mse_lr_poly)
    #     inner_mse_svr_poly.extend(mse_svr_poly)
    
    # np.save("scores/inner_lambda_lr_poly.npy", inner_lambda_lr_poly)
    # np.save("scores/inner_lambda_svr_poly.npy", inner_lambda_svr_poly)
    # np.save("scores/inner_mse_lr_poly.npy", inner_mse_lr_poly)
    # np.save("scores/inner_mse_svr_poly.npy", inner_mse_svr_poly)
    # np.save("scores/n_support_vectors_inner.npy", n_support_vectors_inner)
        
       
    # np.save("scores/mse_lr_poly.npy", mse_lr_poly_nocv)
    # np.save("scores/se_lr_poly.npy", se_lr_poly_nocv)
    # np.save("scores/mse_svr_poly.npy", mse_svr_poly_nocv)
    # np.save("scores/se_svr_poly.npy", se_svr_poly_nocv)
    # np.save("scores/n_support_vectors.npy", n_support_vectors_nocv)
    
    inner_ms_lr_poly = np.load("scores/inner_mse_lr_poly.npy")
    mse_lr_poly = np.load("scores/mse_lr_poly.npy")
    se_lr_poly = np.load("scores/se_lr_poly.npy")
    inner_lambda_lr_poly = np.load("scores/inner_lambda_lr_poly.npy")
    inner_mse_lr_poly_mean = [np.mean(inner_ms_lr_poly[i:i+k]) for i in range(0, len(inner_ms_lr_poly), k)]
    inner_se_lr_poly_se = [np.std(inner_ms_lr_poly[i:i+k]) / np.sqrt(k) for i in range(0, len(inner_ms_lr_poly), k)]
    
    print(inner_lambda_lr_poly)
    print(mse_lr_poly)
    
    plt.plot(inner_mse_lr_poly_mean, label="Ridge Regression, nested CV", c="orange")
    plt.plot(mse_lr_poly, label="Ridge Regression, lambda=1", c="blue")
    plt.errorbar(range(len(inner_mse_lr_poly_mean)), inner_mse_lr_poly_mean, yerr=inner_se_lr_poly_se, fmt='o', c="orange")
    plt.errorbar(range(len(mse_lr_poly)), mse_lr_poly, yerr=se_lr_poly, fmt='o', c="blue")
    plt.legend()
    plt.xlabel("Degree")
    plt.ylabel("MSE")
    plt.yscale("log")
    plt.show()
    print("MSE poly", np.min(mse_lr_poly))
    print("MSE poly inner", np.min(inner_mse_lr_poly_mean))
    
    inner_ms_svr_poly = np.load("scores/inner_mse_svr_poly.npy")
    inner_lambda_svr_poly = np.load("scores/inner_lambda_svr_poly.npy")
    mse_svr_poly = np.load("scores/mse_svr_poly.npy")
    se_svr_poly = np.load("scores/se_svr_poly.npy")
    inner_mse_svr_poly_mean = [np.mean(inner_ms_svr_poly[i:i+k]) for i in range(0, len(inner_ms_svr_poly), k)]
    inner_se_svr_poly_se = [np.std(inner_ms_svr_poly[i:i+k]) / np.sqrt(k) for i in range(0, len(inner_ms_svr_poly), k)]
    print(np.mean(inner_lambda_svr_poly))
    
    print("mse svm poly", np.min(mse_svr_poly))
    print("mse svm poly inner", np.min(inner_ms_svr_poly))
    
    plt.plot(inner_mse_svr_poly_mean, label="SVR, nested CV", c="orange")
    plt.plot(mse_svr_poly, label="SVR, lambda=1", c="blue")
    plt.errorbar(range(len(inner_mse_svr_poly_mean)), inner_mse_svr_poly_mean, yerr=inner_se_svr_poly_se, fmt='o', c="orange")
    plt.errorbar(range(len(mse_svr_poly)), mse_svr_poly, yerr=se_svr_poly, fmt='o', c="blue")
    plt.legend()
    plt.xlabel("Degree")
    plt.ylabel("MSE")
    plt.yscale("log")
    plt.show()
    
    lambda_lr = np.load("scores/inner_lambda_lr_poly.npy")
    lambda_svr = np.load("scores/inner_lambda_svr_poly.npy")
    sv_inner = np.load("scores/n_support_vectors_inner.npy")
    sv_fixed = np.load("scores/n_support_vectors.npy")
    
    mean_split_lambda = np.mean([lambda_lr[i:i+10] for i in range(0, len(lambda_lr), 10)], axis=1)
    mean_split_svr = np.mean([lambda_svr[i:i+10] for i in range(0, len(lambda_svr), 10)], axis=1)
    mean_inner_sv = np.mean([sv_inner[i:i+10] for i in range(0, len(sv_inner), 10)], axis=1)
    print("__________________")
    print("Lambdas (SVR)")
    print(mean_split_svr)
    print(np.mean(mean_split_svr))
    print("Mean SV per split (inner)")
    print(mean_inner_sv)
    print(np.mean(mean_inner_sv))
    print("Fixed lambda SV")
    print(sv_fixed)
    print(np.mean(sv_fixed))
    print("__________________")
    
    # # ############################################### 
    # # # RBF

    # mse_lr_rbf_nocv, se_lr_rbf_nocv = [], []
    # mse_svr_rbf_nocv, se_svr_rbf_nocv = [], []
    # n_support_vectors_nocv = []
    # n_support_vectors_inner = []

    # inner_mse_lr_rbf = []
    # inner_mse_svr_rbf = []
    # inner_lambda_lr_rbf = []
    # inner_lambda_svr_rbf = []


    # for sigma in sigmas:
    #     kfold = KFold(n_splits=10, shuffle=True, random_state=42)
        
    #     # Normal cross validation for M in [1, 10] and lambda=1
    #     print("sigma:", sigma)
    #     cv_lr = cross_validate(KernelizedRidgeRegression(kernel=RBF(sigma=sigma), lambda_=1, threshold=1e-5), X, y, scoring='neg_mean_squared_error', return_estimator=True, n_jobs=-1, cv=kfold)
    #     cv_svr = cross_validate(SVR(kernel=RBF(sigma=sigma), lambda_=1, epsilon=.5, threshold=1e-5), X, y, scoring='neg_mean_squared_error', return_estimator=True, n_jobs=-1, cv=kfold)
        
    #     n_support_vectors_nocv.append(sum([len(est.support_vectors) for est in cv_svr['estimator']]) / 10)
        
    #     mse_lr_rbf_nocv.append(-cv_lr['test_score'].mean())
    #     mse_svr_rbf_nocv.append(-cv_svr['test_score'].mean())
        
    #     se_lr_rbf_nocv.append(cv_lr['test_score'].std() / np.sqrt(10))
    #     se_svr_rbf_nocv.append(cv_svr['test_score'].std() / np.sqrt(10))
        
    #     print("MSE LR RBF:", mse_lr_rbf_nocv[-1], "SE LR RBF:", se_lr_rbf_nocv[-1])
    #     print("MSE SVR RBF:", mse_svr_rbf_nocv[-1], "SE SVR RBF:", se_svr_rbf_nocv[-1])
    #     print("N Support Vectors:", n_support_vectors_nocv[-1])
    #     print("=====================================")

    #     # Nested cross validation for best lambda
    #     mse_lr_rbf = []
    #     mse_svr_rbf = []
    #     lambda_lr_rbf = []
    #     lambda_svr_rbf = []
        
        
    #     for i, (train_ix, test_ix) in enumerate(kfold.split(X)):
    #         best_mse_lr = float("inf")
    #         best_mse_svr = float("inf")
    #         min_sv_count = float("inf")
    #         X_train, X_test = X[train_ix], X[test_ix]
    #         y_train, y_test = y[train_ix], y[test_ix]
            
    #         best_lambda_lr = None
    #         best_lambda_svr = None
            
    #         parameters_lr = {"kernel": [RBF(sigma=sigma)], "lambda_": lambdas, "threshold": [1e-5]}
    #         parameters_svr = {"kernel": [RBF(sigma=sigma)], "lambda_": lambdas, "epsilon":[0.5], "threshold": [1e-5]}
            
    #         ### INNER CV ###
            
    #         gridcv_lr = GridSearchCV(KernelizedRidgeRegression(RBF(sigma=sigma), lambda_=1), parameters_lr, cv=10, scoring='neg_mean_squared_error', n_jobs=-1)
    #         gridcv_svr = GridSearchCV(SVR(RBF(sigma=sigma), lambda_=1, epsilon=0.5), parameters_svr, cv=10, scoring='neg_mean_squared_error', n_jobs=-1)
            
    #         gridcv_lr.fit(X_train, y_train)
    #         gridcv_svr.fit(X_train, y_train)
            
    #         print(f"Best MSE LR: {-gridcv_lr.best_score_}, Best MSE SVR: {-gridcv_svr.best_score_}")
    #         print(f"Best Lambda LR: {gridcv_lr.best_params_['lambda_']}, Best Lambda SVR: {gridcv_svr.best_params_['lambda_']}")
               
    #         if gridcv_lr.best_score_ < best_mse_lr:
    #             best_mse_lr = gridcv_lr.best_score_
    #             best_lambda_lr = gridcv_lr.best_params_['lambda_']
    #             best_model_lr = gridcv_lr.best_estimator_
                
    #         if gridcv_svr.best_score_ < best_mse_svr:
    #             best_mse_svr = gridcv_svr.best_score_
    #             best_lambda_svr = gridcv_svr.best_params_['lambda_']
    #             best_model_svr = gridcv_svr.best_estimator_
                
    #         ### OUTER CV ###
    #         y_pred_lr = best_model_lr.predict(X_test)
    #         y_pred_svr = best_model_svr.predict(X_test)
    #         n_support_vectors_inner.append(len(best_model_svr.support_vectors))
            
    #         mse_lr = mean_squared_error(y_test, y_pred_lr)  
    #         mse_svr = mean_squared_error(y_test, y_pred_svr)
    #         mse_lr_rbf.append(mse_lr)
    #         mse_svr_rbf.append(mse_svr)
    #         lambda_lr_rbf.append(best_lambda_lr)
    #         lambda_svr_rbf.append(best_lambda_svr)
            
    #         print(len(mse_svr_rbf), "MSE LR:", mse_lr, "MSE SVR:", mse_svr)
    #         print(len(lambda_lr_rbf))
            
    #         print(f"Fold {i+1}: Best Lambda LR: {best_lambda_lr} -- MSE LR (with best model): {mse_lr}, Best Lambda SVR: {best_lambda_svr} -- MSE SVR (with best model): {mse_svr}")
    #         print("=====================================")
        
    #     inner_lambda_lr_rbf.extend(lambda_lr_rbf)
    #     inner_lambda_svr_rbf.extend(lambda_svr_rbf)
    #     inner_mse_lr_rbf.extend(mse_lr_rbf)
    #     inner_mse_svr_rbf.extend(mse_svr_rbf)
    
    # np.save("scores/inner_lambda_lr_rbf.npy", inner_lambda_lr_rbf)
    # np.save("scores/inner_lambda_svr_rbf.npy", inner_lambda_svr_rbf)
    # np.save("scores/inner_mse_lr_rbf.npy", inner_mse_lr_rbf)
    # np.save("scores/inner_mse_svr_rbf.npy", inner_mse_svr_rbf)
    # np.save("scores/n_support_vectors_inner_rbf.npy", n_support_vectors_inner)
        
       
    # np.save("scores/mse_lr_rbf.npy", mse_lr_rbf_nocv)
    # np.save("scores/se_lr_rbf.npy", se_lr_rbf_nocv)
    # np.save("scores/mse_svr_rbf.npy", mse_svr_rbf_nocv)
    # np.save("scores/se_svr_rbf.npy", se_svr_rbf_nocv)
    # np.save("scores/n_support_vectors_nocv_rbf.npy", n_support_vectors_nocv)

    inner_ms_lr_rbf = np.load("scores/inner_mse_lr_rbf.npy")
    mse_lr_rbf = np.load("scores/mse_lr_rbf.npy")
    se_lr_rbf = np.load("scores/se_lr_rbf.npy")
    inner_lambda_lr_rbf = np.load("scores/inner_lambda_lr_rbf.npy")
    inner_mse_lr_rbf_mean = [np.mean(inner_ms_lr_rbf[i:i+10]) for i in range(0, len(inner_ms_lr_rbf), 10)]
    inner_se_lr_rbf_se = [np.std(inner_ms_lr_rbf[i:i+10]) / np.sqrt(10) for i in range(0, len(inner_ms_lr_rbf), 10)]
    
    print(inner_lambda_lr_rbf)
    print(mse_lr_rbf)
    
    plt.plot(inner_mse_lr_rbf_mean, label="Ridge Regression, nested CV", c="orange")
    plt.plot(mse_lr_rbf, label="Ridge Regression, lambda=1", c="blue")
    plt.errorbar(range(len(inner_mse_lr_rbf_mean)), inner_mse_lr_rbf_mean, yerr=inner_se_lr_rbf_se, fmt='o', c="orange")
    plt.errorbar(range(len(mse_lr_rbf)), mse_lr_rbf, yerr=se_lr_rbf, fmt='o', c="blue")
    plt.legend()
    plt.xlabel("Sigma")
    plt.xticks(range(len(inner_mse_lr_rbf_mean)), sigmas)
    plt.ylabel("MSE")
    plt.show()
    

    inner_ms_svr_rbf = np.load("scores/inner_mse_svr_rbf.npy")
    inner_lambda_svr_rbf = np.load("scores/inner_lambda_svr_rbf.npy")
    mse_svr_rbf = np.load("scores/mse_svr_rbf.npy")
    se_svr_rbf = np.load("scores/se_svr_rbf.npy")
    inner_mse_svr_rbf_mean = [np.mean(inner_ms_svr_rbf[i:i+10]) for i in range(0, len(inner_ms_svr_rbf), 10)]
    inner_se_svr_rbf_se = [np.std(inner_ms_svr_rbf[i:i+10]) / np.sqrt(10) for i in range(0, len(inner_ms_svr_rbf), 10)]
    print(np.mean(inner_lambda_svr_rbf))
    print("RBF MIN (2x inner 2x outer)")
    print(np.min(inner_mse_lr_rbf_mean))
    print(np.min(inner_mse_svr_rbf_mean))
    print(np.min(mse_lr_rbf))
    print(np.min(mse_svr_rbf))
    
    plt.plot(inner_mse_svr_rbf_mean, label="SVR, nested CV", c="orange")
    plt.plot(mse_svr_rbf, label="SVR, lambda=1", c="blue")
    plt.errorbar(range(len(inner_mse_svr_rbf_mean)), inner_mse_svr_rbf_mean, yerr=inner_se_svr_rbf_se, fmt='o', c="orange")
    plt.errorbar(range(len(mse_svr_rbf)), mse_svr_rbf, yerr=se_svr_rbf, fmt='o', c="blue")
    plt.legend()
    plt.xlabel("Sigma")
    plt.xticks(range(len(inner_mse_lr_rbf_mean)), sigmas)
    plt.yscale("log")
    plt.ylabel("MSE")
    plt.show()

    lambda_lr = np.load("scores/inner_lambda_lr_rbf.npy")
    lambda_svr = np.load("scores/inner_lambda_svr_rbf.npy")
    sv_inner = np.load("scores/n_support_vectors_inner_rbf.npy")
    sv_fixed = np.load("scores/n_support_vectors_nocv_rbf.npy")
    
    mean_split_lambda = np.mean([lambda_lr[i:i+10] for i in range(0, len(lambda_lr), 10)], axis=1)
    mean_split_svr = np.mean([lambda_svr[i:i+10] for i in range(0, len(lambda_svr), 10)], axis=1)
    mean_inner_sv = np.mean([sv_inner[i:i+10] for i in range(0, len(sv_inner), 10)], axis=1)
    print("__________________")
    print("Lambdas (SVR)")
    print(mean_split_svr)
    print(np.mean(mean_split_svr))
    print("Mean SV per split (inner)")
    print(mean_inner_sv)
    print(np.mean(mean_inner_sv))
    print("Fixed lambda SV")
    print(sv_fixed)
    print(np.mean(sv_fixed))
    print("__________________")
