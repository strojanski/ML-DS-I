import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from collections import Counter

categorical = ["ShotType", "Competition", "PlayerType", "Movement"]
eps = 1e-9


def sigmoid(X):
    return 1 / (1 + np.exp(-np.clip(X, -500, 500)))

def dx_sigmoid(X, y):
    return sigmoid(X) * (1 - sigmoid(X))

def softmax(X):
    exp_X = np.exp(X - np.max(X, axis=1, keepdims=True))
    return exp_X / np.sum(exp_X, axis=1, keepdims=True)


np.random.seed(1)
class_names_orig = []
class MultinomialLogReg:
    def __init__(self):
        self.opt_theta = None
        self.n_classes = None

    def log_likelihood(self, theta, X, y):
        theta = theta.reshape((self.n_feats, self.n_classes - 1))  # Reshape to (n_feats, m-1)
        theta_full = np.hstack([theta, np.zeros((self.n_feats, 1))])  # Add zero-coeff class

        logits = X @ theta_full
        probs = softmax(logits)

        y_onehot = np.zeros_like(probs)
        y_onehot[np.arange(len(y)), y] = 1
        
        return -np.sum(y_onehot * np.log(probs + eps))  # Avoid log(0)
    
    def gradient(self, theta, X, y):
        theta = theta.reshape((X.shape[1], self.n_classes - 1))
        theta_full = np.hstack([theta, np.zeros((self.n_feats, 1))])  # Add zero-coeff class

        logits = X @ theta_full
        probs = softmax(logits)
        
        y_onehot = np.zeros_like(probs)
        y_onehot[np.arange(len(y)), y] = 1
        
        grad = (X.T @ (probs - y_onehot))
        return grad[:, :-1].ravel()
    
    def build(self, X, y):
        self.n_classes = len(np.unique(y))
        self.n_feats = X.shape[1]
        initial_guess = np.random.randn(self.n_feats * (self.n_classes - 1)) / 100
               
        opt_theta, f, d = fmin_l_bfgs_b(self.log_likelihood, initial_guess, args=(X, y), fprime=self.gradient)
        
        self.opt_theta = np.hstack([opt_theta.reshape((self.n_feats, self.n_classes - 1)), np.zeros((self.n_feats, 1))])  # Reconstruct full matrix
        
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
        thresholds = np.sort(params[self.n_feats:]) # Cutoffs - must be ordered
        
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
        
        initial_guess_feats = np.random.randn(self.n_feats)
        initial_guess_thresh = np.sort(np.random.rand(self.n_thresh))
        initial_guess = np.concatenate([initial_guess_feats, initial_guess_thresh])

        opt_params, f, d = fmin_l_bfgs_b(self.log_likelihood, initial_guess, args=(X, y), fprime=self.gradient)
        self.theta = opt_params[:self.n_feats].reshape((self.n_feats, 1))  # Ensure theta is a column vector
        self.thresh = np.sort(opt_params[self.n_feats:])
        
        return self
    
    def predict(self, X):
        logits = X @ self.theta
        probs = self.sigmoid_cdf(logits, self.thresh)
        return probs

def plot_predicted_probs(model, X, feature_idx, feature_name, class_labels):
    """Plots predicted probabilities as a function of one feature"""
    feature_values = np.linspace(X[:, feature_idx].min(), X[:, feature_idx].max(), 100)
    X_test = np.mean(X, axis=0, keepdims=True).repeat(100, axis=0)
    X_test[:, feature_idx] = feature_values
    
    probs = model.predict(X_test)  # Get predicted probabilities
    
    plt.figure(figsize=(8, 5))
    for i in range(probs.shape[1]):
        plt.plot(feature_values, probs[:, i], label=class_labels[i])
    
    plt.xlabel(feature_name)
    plt.ylabel("Predicted Probability")
    plt.title(f"Effect of {feature_name} on Shot Type")
    plt.legend()
    plt.show()
    

def plot_coefficients(model, feature_names, thetas_var):
    """
    Plots the odds ratios (exp(coefficients)) from a trained multinomial logistic regression model
    with confidence intervals from bootstrapped variance.

    Args:
        model: Trained MultinomialLogReg model.
        feature_names: List of feature names.
        thetas_var: Bootstrapped variance of the coefficients (shape: n_features × n_classes).
    """
    theta_means = model.opt_theta  # Extract learned coefficients
    theta_std = np.sqrt(thetas_var)  # Compute standard deviation (square root of variance)

    odds_ratios = np.exp(theta_means)  # Convert to odds ratios
    error_bars = np.exp(theta_means + theta_std) - np.exp(theta_means)  # Compute confidence intervals

    num_classes = theta_means.shape[1]
    num_features = len(feature_names)
    
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray']
    markers = ['o', 's', '^', 'D', 'P', '*', 'X', 'v']
    
    plt.figure(figsize=(12, 6))

    for class_idx in range(num_classes):
        # Jitter x positions slightly to prevent overlap
        x_jitter = np.arange(num_features) + np.random.uniform(-0.2, 0.2, size=num_features)
        
        plt.errorbar(
            x=x_jitter, 
            y=odds_ratios[:, class_idx], 
            yerr=error_bars[:, class_idx], 
            capsize=6, capthick=2, elinewidth=1.5,
            fmt=markers[class_idx % len(markers)], 
            color=colors[class_idx % len(colors)],
            label=f'{class_names_orig[class_idx]}',
            alpha=0.7
        )

    plt.xticks(ticks=np.arange(num_features), labels=feature_names, rotation=45, ha='right', fontsize=12)
    plt.yscale('log')  # Use log scale for odds ratios
    plt.ylabel("Odds Ratio", fontsize=14)
    
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=12)
    plt.tight_layout()  # Ensures labels and legends don't get cut off
    
    plt.show()


def bootstrap_thetas(model, X, y, n_bootstrap=100):
    """Estimates standard errors using bootstrap resampling."""
    n_samples = X.shape[0]
    bootstrap_coefs = []

    for _ in range(n_bootstrap):
        indices = np.random.choice(n_samples, n_samples, replace=True)
        X_boot, y_boot = X[indices], y[indices]
        
        model = model.build(X_boot, y_boot)
        bootstrap_coefs.append(model.opt_theta)
    
    return np.std(np.array(bootstrap_coefs), axis=0)  


def permutation_importance(model, X, y, metric):
    baseline = metric(y, np.argmax(model.predict(X), axis=1)) 
    
    importances = []
    
    for i in range(X.shape[1]):
        X_per = X.copy()
        X_per[:, i] = np.random.permutation(X_per[:,i])
        
        score_per = metric(y, np.argmax(model.predict(X_per), axis=1))
        
        imp = baseline - score_per
        importances.append((i, imp))
        
    importances = sorted(importances, key=lambda x: x[1], reverse=True)
    
    return importances
    
def plot_perm_imps(imps):
    feats_ = [i[0] for i in imps]
    imps_ = [i[1] for i in imps]
    
    plt.bar(feats[feats_], imps_)
    plt.xticks(rotation=30)
    plt.ylabel("Importance")
    plt.xlabel("Feature name")
    plt.show()
    print(imps)
    
    
def plot_residuals(model, X, y):
    # Step 1: Get predicted probabilities (shape: n_samples x n_classes)
    probs = model.predict(X)
    
    # Step 2: Calculate residuals for each sample
    # Residual = predicted probability for the true class - 1
    residuals = probs[np.arange(len(y)), y] - 1  # Get the predicted probability for the true class and subtract 1
    
    # Step 3: Plot the residuals
    plt.figure(figsize=(10, 6))
    plt.scatter(np.arange(len(residuals)), residuals, color='red', alpha=0.6)
    
    # Add horizontal line at residual = 0 (perfect prediction)
    plt.axhline(0, color='black', linewidth=1, linestyle='--')  
    
    # Add labels and title
    plt.xlabel('Sample Index')
    plt.ylabel('Residuals (True - Predicted Probability)')
    plt.title('Model Residuals')
    
    # Show the plot
    plt.show()


def multinomial_bad_ordinal_good(n_samples=1000, n_features=3, random_seed=42):
    """Create a DGP with ordinal relationship"""
    np.random.seed(random_seed)
    
    X = np.random.randn(n_samples, n_features) 
    beta = np.array([0.5, -0.3, 0.8])
    # beta = np.random.rand(n_features) 
    noise = np.random.randn(n_samples)  
    y_star = X @ beta + noise  
    
    thresholds = [0, 1]
    
    y = np.digitize(y_star, thresholds) # Make it ordinal
    
    return X, y

if __name__ == "__main__":
    df = pd.read_csv("C:/Users/sebas/one/OneDrive/grive/faks/masters/y1/2nd semester/ML-DS I/ML-DS-I/hw3/dataset.csv", sep=";")
    le = LabelEncoder()
    print(multinomial_bad_ordinal_good())
    
    # Encode labels
    original_feat_names = []
    for col in categorical:
        df[col] = le.fit_transform(df[col])
        print(le.classes_)
        original_feat_names.append(le.classes_)
    class_names_orig = original_feat_names[0]
    
    # Get data
    X, y = df.drop(columns=["ShotType"]), df["ShotType"] 
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, stratify=y)
    
    X_train, y_train = X_train.to_numpy(), y_train.to_numpy()
    X_test, y_test = X_test.to_numpy(), y_test.to_numpy()
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)


    # Multinomial LR
    lr = MultinomialLogReg()
    lr = lr.build(X_train, y_train)
    
    preds = lr.predict(X_test)
    feats = np.array(["Competition", "PlayerType", "Transition", "TwoLegged", "Movement", "Angle", "Distance"])
    
    # Probability vs. feature values
    plot_predicted_probs(lr, X.to_numpy(), feature_idx=1, feature_name="PlayerType", class_labels=original_feat_names[0])
    
    # Odds ratio vs features vs classes
    thetas_var = bootstrap_thetas(lr, X.to_numpy(), y.to_numpy(), n_bootstrap=1)
    plot_coefficients(lr, feats, thetas_var)
    
    # Permutation based importances
    imps = permutation_importance(lr, X_test, y_test, accuracy_score)
    plot_perm_imps(imps)
    
    
    preds = np.argmax(preds, axis=1)

    print(classification_report(y_test, preds, zero_division=0.0))
    # print(accuracy_score(y_test, preds))

    # print(accuracy_score(y_test, preds))
    
    X, y = multinomial_bad_ordinal_good()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, stratify=y)
    
    lr.build(X_train, y_train)
    
    preds = lr.predict(X_test)
    
    print(accuracy_score(y_test, np.argmax(preds, axis=1)))
    
    
    olr = OrdinalLogReg()
    olr = olr.build(X_train, y_train)
    preds = olr.predict(X_test)
    print(accuracy_score(y_test, np.argmax(preds, axis=1)))
    
    # preds = olr.predict(X_test)
    # preds = np.argmax(preds, axis=1)
    # print(classification_report(y_test, preds, zero_division=0.0))
    # print(accuracy_score(y_test, preds))

