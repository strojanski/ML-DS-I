import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns

categorical = ["ShotType", "Competition", "PlayerType", "Movement"]
eps = 1e-9
np.random.seed(1)
class_names_orig = []

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
        self.n_feats = None

    def log_likelihood(self, theta, X, y):
        theta = theta.reshape((self.n_feats, self.n_classes - 1))
        theta_full = np.hstack([theta, np.zeros((self.n_feats, 1))])  # Fix last class weights to 0

        logits = X @ theta_full
        probs = softmax(logits)

        y_onehot = np.zeros_like(probs)
        y_onehot[np.arange(len(y)), y] = 1

        return -np.sum(y_onehot * np.log(probs + eps))  

    def gradient(self, theta, X, y):
        theta = theta.reshape((self.n_feats, self.n_classes - 1))
        theta_full = np.hstack([theta, np.zeros((self.n_feats, 1))])  # Fix last class weights to 0

        logits = X @ theta_full
        probs = softmax(logits)

        y_onehot = np.zeros_like(probs)
        y_onehot[np.arange(len(y)), y] = 1

        grad_full = X.T @ (probs - y_onehot)
        grad = grad_full[:, :self.n_classes - 1]  # Exclude gradient for fixed class

        return grad.ravel()

    def fit(self, X, y):
        return self.build(X, y)

    def build(self, X, y):
        self.n_classes = len(np.unique(y))
        self.n_feats = X.shape[1]
        initial_guess = np.random.randn(self.n_feats * (self.n_classes - 1)) / 100

        opt_theta, _, _ = fmin_l_bfgs_b(
            self.log_likelihood,
            initial_guess,
            args=(X, y),
            fprime=self.gradient,
            bounds=None,
            maxiter=10000
        )

        theta = opt_theta.reshape((self.n_feats, self.n_classes - 1))
        self.opt_theta = np.hstack([theta, np.zeros((self.n_feats, 1))])
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
        theta = params[:self.n_feats].reshape((self.n_feats, 1))
        thresholds = np.sort(params[self.n_feats:])
        
        # Clip logits to prevent overflow in sigmoid
        logits = np.clip(X @ theta, -20, 20)
        probs = self.sigmoid_cdf(logits, thresholds)
        
        # Numerical stability: avoid log(0) and use log-sum-exp trick
        log_probs = np.log(probs[np.arange(len(y)), y] + 1e-15)
        
        return -np.sum(log_probs)

    def sigmoid_cdf(self, logits, thresholds):
        n_samples = logits.shape[0]
        n_thresh = len(thresholds)
        cum_probs = np.zeros((n_samples, n_thresh + 1))
        
        # First threshold: P(Y ≤ 0)
        cum_probs[:, 0] = sigmoid(thresholds[0] - logits.ravel())
        
        # Middle thresholds: P(Y ≤ k) - P(Y ≤ k-1)
        for j in range(1, n_thresh):
            cum_probs[:, j] = sigmoid(thresholds[j] - logits.ravel()) - sigmoid(thresholds[j-1] - logits.ravel())
        
        # Last class: 1 - P(Y ≤ K-1)
        cum_probs[:, -1] = 1 - sigmoid(thresholds[-1] - logits.ravel())
        
        # Ensure valid probabilities (sum to 1)
        cum_probs = np.clip(cum_probs, eps, 1-eps)
        return cum_probs / cum_probs.sum(axis=1, keepdims=True)  # Renormalize

    def gradient(self, params, X, y):
        theta = params[:self.n_feats].reshape((self.n_feats, 1))
        thresholds = np.sort(params[self.n_feats:])
        n_samples, n_thresh = X.shape[0], len(thresholds)
        
        logits = np.clip(X @ theta, -20, 20).ravel()
        
        # Compute cumulative probabilities P(Y <= j)
        cum_probs = np.zeros((n_samples, n_thresh + 1))
        cum_probs[:, 0] = sigmoid(thresholds[0] - logits)
        for j in range(1, n_thresh):
            cum_probs[:, j] = sigmoid(thresholds[j] - logits)
        cum_probs[:, -1] = 1.0
        
        # Indicators I(y <= j) for j=0,...,n_thresh-1
        indicators = np.zeros((n_samples, n_thresh))
        for j in range(n_thresh):
            indicators[:, j] = (y <= j)
        
        # Gradient for theta
        grad_theta = X.T @ (indicators - cum_probs[:, :n_thresh]).sum(axis=1)
        
        # Gradient for thresholds
        grad_thresh = np.sum(cum_probs[:, :n_thresh] - indicators, axis=0)
        
        return np.concatenate([grad_theta.ravel(), grad_thresh.ravel()])

    def fit(self, X, y):
        return self.build(X, y)
    
    def build(self, X, y):
        self.n_feats = X.shape[1]
        unique_y = np.unique(y)
        self.n_thresh = len(unique_y) - 1
        
        # Initialize thresholds using empirical class distribution
        class_counts = np.bincount(y)
        class_probs = class_counts / class_counts.sum()
        cum_probs = np.cumsum(class_probs)[:-1]
        cum_probs = np.clip(cum_probs, 0.01, 0.99)  # Avoid log(0)
        initial_guess_thresh = np.log(cum_probs / (1 - cum_probs))  # Logit
        
        # Initialize theta with zeros or small random values
        initial_guess_feats = np.zeros(self.n_feats) 
        initial_guess = np.concatenate([initial_guess_feats, initial_guess_thresh])
        
        opt_params = fmin_l_bfgs_b(self.log_likelihood, initial_guess, 
                                fprime=self.gradient, args=(X, y))[0]
        self.theta = opt_params[:self.n_feats].reshape((self.n_feats, 1))
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
    
    plt.xlabel(feature_name, fontsize=14)
    plt.ylabel("Predicted Probability", fontsize=14)
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
    theta_means = model.opt_theta  
    theta_std = np.sqrt(thetas_var)  

    # Compute odds ratios for the means and the standard deviation
    odds_ratios = np.exp(theta_means)  
    error_bars_lower = np.exp(theta_means - 1.96 * theta_std)  
    error_bars_upper = np.exp(theta_means + 1.96 * theta_std)  

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
            yerr=[odds_ratios[:, class_idx] - error_bars_lower[:, class_idx], 
                  error_bars_upper[:, class_idx] - odds_ratios[:, class_idx]],
            capsize=6, capthick=2, elinewidth=1.5,
            fmt=markers[class_idx % len(markers)], 
            color=colors[class_idx % len(colors)],
            label=f'{class_names_orig[class_idx]}',
            alpha=0.7
        )

    plt.xticks(ticks=np.arange(num_features), labels=feature_names, rotation=45, ha='right', fontsize=12)
    plt.yscale('log')  
    plt.ylabel("Odds Ratio", fontsize=14)
    plt.axhline(y=1, color='gray', linestyle='--')  
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=12)
    plt.tight_layout()  
    
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
    plt.ylabel("Importance", fontsize=14)
    plt.xlabel("Feature name", fontsize=14)
    plt.show()
    print(imps)
    
    
def plot_residuals(model, X, y):
    probs = model.predict(X)
    
    residuals = probs[np.arange(len(y)), y] - 1  
    
    plt.figure(figsize=(10, 6))
    plt.scatter(np.arange(len(residuals)), residuals, color='red', alpha=0.6)
    
    plt.axhline(0, color='black', linewidth=1, linestyle='--')  
    
    plt.xlabel('Sample Index')
    plt.ylabel('Residuals (True - Predicted Probability)')
    plt.title('Model Residuals')
    
    plt.show()


def multinomial_bad_ordinal_good(n_samples=1000, n_features=6, n_classes=10, random_seed=42):
    """Create a DGP with ordinal relationship"""
    np.random.seed(random_seed)
    X = np.random.randn(n_samples, n_features)
    
    # Define coefficients and generate a latent variable y*
    beta = np.random.randn(n_features)
    noise = np.random.randn(n_samples)
    y_star = X @ beta + (noise / 100)
    
    thresholds = np.percentile(y_star, np.linspace(0, 100, n_classes + 1)[1:-1])
    
    y = np.digitize(y_star, thresholds) # Make it ordinal
    
    return X, y

from matplotlib.colors import ListedColormap

def plot_decision_boundary(model, X, y):
    h = 0.1
    x_min, x_max = X[:, 0].min()-1, X[:, 0].max()+1
    y_min, y_max = X[:, 1].min()-1, X[:, 1].max()+1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    grid = np.c_[xx.ravel(), yy.ravel()]
    probs = model.predict(grid)
    Z = np.argmax(probs, axis=1).reshape(xx.shape)

    cmap = ListedColormap(['#AAAAFF', '#FFAAAA', '#AAFFAA'])
    plt.contourf(xx, yy, Z, cmap=cmap, alpha=0.5)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap='bwr')
    plt.show()
    
def plot_coefficient_heatmap(model, feature_names, class_names):
    theta_means = model.opt_theta
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        theta_means.T,
        annot=True,
        fmt=".2f",
        cmap="RdBu",
        center=0,
        xticklabels=feature_names,
        yticklabels=class_names
    )
    # plt.title("Log-Odds Coefficients Heatmap")
    plt.tight_layout()
    plt.show()
    
    
if __name__ == "__main__":
    df = pd.read_csv("dataset.csv", sep=";")
    le = LabelEncoder()
    

    
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

    # Mock data for plot
    X2 = np.array([
            [1,.7],
            [.1,-2],
            [1,0],
            [2,0],
            [1,0],
            [2,0],
            [4,5],
            [3,3],
            [4,4],
            [3,5]]
            )
    y2 = np.array([0,0,0,0,0,0,1,1,1,1])
    plot_decision_boundary(lr.build(X2, y2), X2, y2)

    # 10 times repeated 5 fold CV
    # accs, f1s = [], []
    # n_rows = len(X) // 10  # use integer division
    # for _ in range(10):
    #     Xy = X.copy()
    #     Xy['label'] = y
    #     Xy = Xy.sample(frac=1).reset_index(drop=True)
    #     for i in range(5):
    #         X_test = Xy.iloc[i*n_rows:(i+1)*n_rows]
    #         X_train = Xy.drop(X_test.index)

    #         y_test = X_test['label']
    #         X_test = X_test.drop(columns='label')

    #         y_train = X_train['label']
    #         X_train = X_train.drop(columns='label')
            
    #         lr.build(X_train.to_numpy(), y_train.to_numpy())
    #         preds = lr.predict(X_test.to_numpy())
            
    #         acc = accuracy_score(y_test.to_numpy(), np.argmax(preds, axis=1))
    #         f1 = f1_score(y_test.to_numpy(), np.argmax(preds, axis=1), average="macro")
    #         accs.append(acc)
    #         f1s.append(f1)
    #         print(acc, f1)
    
    # print(np.mean(accs), np.std(accs))
    # print(np.mean(f1s), np.std(f1s))       
        

    # lr = lr.build(X_train, y_train)
    
    # # Visualizations
    # preds = lr.predict(X_test)
    # feats = np.array(["Competition", "PlayerType", "Transition", "TwoLegged", "Movement", "Angle", "Distance"])
    
    # # Heatmap
    # plot_coefficient_heatmap(lr, feature_names=feats, class_names=class_names_orig)

    # # Probability vs. feature values
    # plot_predicted_probs(lr, X.to_numpy(), feature_idx=-1, feature_name="Distance", class_labels=original_feat_names[0])
    
    # # Odds ratio vs features vs classes
    # thetas_var = bootstrap_thetas(lr, X.to_numpy(), y.to_numpy(), n_bootstrap=100)
    # plot_coefficients(lr, feats, thetas_var)
    
    # # Permutation based importances
    # imps = permutation_importance(lr, X_test, y_test, accuracy_score)
    # plot_perm_imps(imps)
    
    # preds = np.argmax(preds, axis=1)
    # print(classification_report(y_test, preds, zero_division=0.0))


    accsl, accso = [], []
    for i in range(100):
        X, y = multinomial_bad_ordinal_good(random_seed=i)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, stratify=y)
        lr = MultinomialLogReg()
        lr.build(X_train, y_train)
        
        preds = lr.predict(X_test)
        
        print("Multinomial:", accuracy_score(y_test, np.argmax(preds, axis=1)))
        accsl.append(accuracy_score(y_test, np.argmax(preds, axis=1)))
        
        olr = OrdinalLogReg()
        olr = olr.build(X_train, y_train)
        preds = olr.predict(X_test)
        print("Ordinal:", accuracy_score(y_test, np.argmax(preds, axis=1)))
        accso.append(accuracy_score(y_test, np.argmax(preds, axis=1)))
    
    print(np.mean(accsl), np.std(accsl))
    print(np.mean(accso), np.std(accso))
    
