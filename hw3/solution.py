import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

categorical = ["ShotType", "Competition", "PlayerType", "Movement"]
eps = 1e-9


def sigmoid(X):
    return 1 / (1 + np.exp(-np.clip(X, -500, 500)))

def dx_sigmoid(X, y):
    return sigmoid(X) * (1 - sigmoid(X))

def softmax(X):
    exp_X = np.exp(X - np.max(X, axis=1, keepdims=True))
    return exp_X / np.sum(exp_X, axis=1, keepdims=True)

class MultinomialLogReg:
    def __init__(self):
        self.opt_theta = None
        self.n_classes = None

    def log_likelihood(self, theta, X, y):
        theta = theta.reshape((X.shape[1], self.n_classes))
        
        logits = X @ theta
        probs = softmax(logits)
        
        y_onehot = np.zeros_like(probs)
        y_onehot[np.arange(len(y)), y] = 1
        
        return -np.sum(y_onehot * np.log(probs + eps))
    
    def gradient(self, theta, X, y):
        theta = theta.reshape((X.shape[1], self.n_classes))
        
        logits = X @ theta
        probs = softmax(logits)
        
        y_onehot = np.zeros_like(probs)
        y_onehot[np.arange(len(y)), y] = 1
        
        return (X.T @ (probs - y_onehot)).ravel()
    
    def build(self, X, y):
        self.n_classes = len(np.unique(y))
        n_feats = X.shape[1]
        initial_guess = np.random.randn(n_feats * self.n_classes) / 100
               
        opt_theta, f, d = fmin_l_bfgs_b(self.log_likelihood, initial_guess, args=(X, y), fprime=self.gradient)
        
        print(f)
        
        self.opt_theta = opt_theta.reshape((n_feats, self.n_classes))
        
        return self
    
    def predict(self, X):
        logits = X @ self.opt_theta
        probs = softmax(logits)
        return probs
        # return np.argmax(probs, axis=1)
    
class OrdinalLogReg():
    def __init__(self):
        self.theta = None
        self.thresh = None
        self.n_feats = None
        self.n_thresh = None
    
    def log_likelihood(self, params, X, y):
        # Sigmoid CDF
        theta = params[:self.n_feats].reshape((self.n_feats, 1))  # Ensure theta is a column vector
        thresholds = np.sort(params[self.n_feats:])
        
        logits = X @ theta
        probs = self.sigmoid_cdf(logits, thresholds)
        
        return -np.sum(np.log(probs[np.arange(len(y)), y] + eps))
    
    def sigmoid_cdf(self, logits, thresholds):
        cum_probs = np.zeros((logits.shape[0], len(thresholds) + 1))
        cum_probs[:, 0] = sigmoid(thresholds[0] - logits).ravel()
        cum_probs[:, -1] = 1 - sigmoid(thresholds[-1] - logits).ravel()
    
        for j in range(1, len(thresholds)):
            cum_probs[:, j] = sigmoid(thresholds[j] - logits).ravel() - sigmoid(thresholds[j - 1] - logits).ravel()

        return cum_probs

    def gradient(self, params, X, y):
        # Get thetas and thresholds
        theta = params[:self.n_feats].reshape((self.n_feats, 1)) # Feature weights
        thresholds = np.sort(params[self.n_feats:]) # Cutoffs
        
        # Get probabilities
        logits = X @ theta 
        probs = self.sigmoid_cdf(logits, thresholds)  
        
        y_onehot = np.zeros_like(probs)
        y_onehot[np.arange(len(y)), y] = 1 
        
        # Compute gradient wrt. theta
        grad_theta = X.T @ (probs - y_onehot) 
        grad_theta = grad_theta.sum(axis=1)[:-1]
        
        # Compute gradient wrt. threshold
        grad_thresh = np.sum(probs - y_onehot, axis=0)  
        return np.concatenate([grad_theta.ravel(), grad_thresh.ravel()])

    
    def build(self, X, y):
        self.n_feats = X.shape[1]
        self.n_thresh = len(np.unique(y)) - 1
        
        initial_guess = np.random.randn(self.n_feats + self.n_thresh)   # Optimize feature thetas & thresholds
        print(initial_guess.shape)

        opt_params, f, d = fmin_l_bfgs_b(self.log_likelihood, initial_guess, args=(X, y), fprime=self.gradient)
        self.theta = opt_params[:self.n_feats].reshape((self.n_feats, 1))  # Ensure theta is a column vector
        self.thresh = np.sort(opt_params[self.n_feats:])
        
        return self
    
    def predict(self, X):
        logits = X @ self.theta
        probs = self.sigmoid_cdf(logits, self.thresh)
        return probs



if __name__ == "__main__":
    
    df = pd.read_csv("C:/Users/sebas/one/OneDrive/grive/faks/masters/y1/2nd semester/ML-DS I/ML-DS-I/hw3/dataset.csv", sep=";")
    le = LabelEncoder()
    
    for col in categorical:
        df[col] = le.fit_transform(df[col])
    
    X, y = df.drop(columns=["ShotType"]), df["ShotType"]    
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)
    
    X_train, y_train = X_train.to_numpy(), y_train.to_numpy()
    X_test, y_test = X_test.to_numpy(), y_test.to_numpy()
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)


    # lr = MultinomialLogReg()
    # lr = lr.build(X_train, y_train)
    
    # preds = lr.predict(X_test)
    # preds = np.argmax(preds, axis=1)
    # print(accuracy_score(y_test, preds))
    
    olr = OrdinalLogReg()
    olr = olr.build(X_train, y_train)
    
    preds = olr.predict(X_test)
    preds = np.argmax(preds, axis=1)
    print(accuracy_score(y_test, preds))
