import pandas as pd
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from collections import Counter
import warnings
from sklearn.exceptions import ConvergenceWarning
import os
import time

os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")


RANDOM_SEED = 0
TARGET = "ShotType"
categorical_cols = ["Competition", "PlayerType", "Movement"]
regularization_params = [1e-2, 1e-1, 1]
kernels = ["linear", "rbf"]


def log_score(y_true, y_pred, epsilon=1e-15):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    
    num_classes = y_pred.shape[1]
    y_true_one_hot = np.eye(num_classes)[y_true]
    
    # Compute log loss
    log_loss_value = -np.mean(np.sum(y_true_one_hot * np.log(y_pred), axis=1))
    
    return - log_loss_value
    
def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def load_data():
    return pd.read_csv("dataset.csv", sep=";")

def baseline():
    dc = DummyClassifier(strategy="stratified", random_state=RANDOM_SEED)
    return dc

def logistic_regression():
    return LogisticRegression(solver="lbfgs", n_jobs=-1, max_iter=500, random_state=RANDOM_SEED)

def svc(C, max_iter=None, kernel="linear", gamma="scale"):
    return SVC(C=C, kernel=kernel, random_state=RANDOM_SEED, max_iter=max_iter, probability=True, gamma=gamma)


def get_folds(df, k=10):
    # Shuffle
    _df = df.copy().sample(frac=1)
    
    fold_size = len(_df) // k
    
    folds = [(i * fold_size, min((i+1) * fold_size, len(_df))) for i in range(k)]
    
    return _df, folds


def get_stratified_folds(df, target_column, k=10, shuffle=True):
    # Shuffle data and get the target column
    _df = df.copy()
    if shuffle:
        _df = _df.sample(frac=1, random_state=RANDOM_SEED)  # Shuffle the dataset
    
    # Group by the target column (y_true), then split each group into stratified folds
    target_classes = _df[target_column].unique()
    
    folds = [[] for _ in range(k)]  # Create k empty folds

    # For each class, stratify the data into k folds
    for class_label in target_classes:
        class_data = _df[_df[target_column] == class_label]
        
        # Split the class data into k equally sized (or nearly equal) parts
        fold_size = len(class_data) // k
        for i in range(k):
            fold_start = i * fold_size
            fold_end = (i + 1) * fold_size if i != k-1 else len(class_data)
            folds[i].extend(class_data.iloc[fold_start:fold_end].index.tolist())
    
    # Convert the list of indices into fold ranges
    stratified_folds = [(min(fold), max(fold)) for fold in folds]
    
    return _df, stratified_folds

def get_train_test_folds(df, fold, i):
    test_start, test_end = fold
    test = df[test_start:test_end].reset_index()

    # Train indices: concatenate all folds except the current one
    train_folds = [f for j, f in enumerate(folds) if j != i]
    train_indices = []
    for f in train_folds:
        train_indices.extend(range(f[0], f[1]))
    train = df.iloc[train_indices].reset_index()
    
    return train, test

def cross_validate(df, folds, model, svm=False):
    
    scores = []
    
    errors_, distances_, comps_ = [], [], []
    
    for i, fold in enumerate(folds):
        
        train, test = get_train_test_folds(df, fold, i)

        X_train, y_train = train.drop(columns=TARGET), train[TARGET]
        X_test, y_test = test.drop(columns=TARGET), test[TARGET]
        
        X_train, X_test = scale_data(X_train, X_test)
        
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        preds_proba = model.predict_proba(X_test)
        logscore = log_score(y_test, preds_proba)
        
        
        true_class_indices = y_test.astype(int)  # Convert true class labels to integers

        # Compute log loss for each data point
        log_loss_per_sample = - (true_class_indices * np.log(preds_proba[np.arange(len(y_test)), true_class_indices]) + 
                                (1 - true_class_indices) * np.log(1 - preds_proba[np.arange(len(y_test)), true_class_indices]))
        
        log_loss_per_sample *= -1

        # Get distances for the test set
        distances = test["Distance"].to_numpy()
        
        # Store (error, distance) pairs
        errors_.append(list(log_loss_per_sample))
        distances_.append(list(distances))
        comps_.append(list(test["Competition"].to_numpy()))
        
        
        acc = accuracy(y_test, preds)
        scores.append((logscore, acc))
    
    return scores, (errors_, distances_, comps_)

def cross_validate_train_optimization(df, folds):
    
    logscores = []
    accs = []
    
    
    for i, fold in enumerate(folds):
        best_log = -np.inf
        best_c = None
        best_kernel = None
        best_gamma = None
        best_accuracy = None
        tmp_logs = []
        
        train, test = get_train_test_folds(df, fold, i)
        X_train, y_train = train.drop(columns=[TARGET]), train[TARGET]
        X_test, y_test = test.drop(columns=[TARGET]), test[TARGET]

        X_train, X_test = scale_data(X_train, X_test)
        for C in regularization_params:
            for kernel in kernels:
                gammas = ["scale"]
                if kernel == "rbf":
                    gammas = [1e-2, 1e-1, 1, "scale"]
                
                for gamma in gammas:
                    model = svc(C=C, max_iter=200, kernel=kernel, gamma=gamma)

                    model.fit(X_train, y_train)
                    
                    preds = model.predict(X_train)
                    pred_probs = model.predict_proba(X_train)
                    
                    logscore = log_score(y_train, pred_probs)
                    acc = accuracy(y_train, preds)
                    
                    tmp_logs.append(logscore)
                    # accs.append(acc)
                    
                    print(f"C: {C}, kernel: {kernel} gamma: {gamma} :: log score: {logscore}, accuracy: {acc}")
                    
                mean_log = np.mean(tmp_logs)
                
                # TODO account for accuracy
                if mean_log > best_log:
                    best_log = mean_log
                    best_kernel = kernel
                    best_gamma = gamma
                    best_c = C


        print("Fold number and best params: ", i, best_c, best_kernel, best_log)
        model = svc(C=best_c, max_iter=200, kernel=best_kernel, gamma=best_gamma)
        
        model.fit(X_train, y_train)
        
        preds = model.predict(X_test)
        pred_probs = model.predict_proba(X_test)
        
        logscore = log_score(y_test, pred_probs)
        acc = accuracy(y_test, preds)
        
        logscores.append(logscore)
        accs.append(acc)
        
    final_log, (log_lb, log_ub) = bootstrap_metric(logscores)
    final_acc, (acc_lb, acc_ub) = bootstrap_metric(accs)
    print(f"\nFinal Nested CV Log Score: {final_log:.4f} (CI: [{log_lb:.4f}, {log_ub:.4f}])")
    print(f"Final Nested CV Accuracy: {final_acc:.4f} (CI: [{acc_lb:.4f}, {acc_ub:.4f}])")
    
            
    return best_c, best_kernel, best_gamma, logscores, accs
        
def nested_cross_validation(df, folds):
    scores = []
    
    # Start outer CV
    for i, fold in enumerate(folds):
        train, test = get_train_test_folds(df, fold, i)
        
        _, folds2 = get_stratified_folds(df, TARGET, k=5, shuffle=True)
        
        # Inner CV - hyperparameter tuning
        best_log = -np.inf
        best_c = None
        best_kernel = None
        best_gamma = None
        
        # Start inner CV
        for C in regularization_params:
            for kernel in kernels:
                gammas = ["scale"]
                if kernel == "rbf":
                    gammas = [1e-3, 1e-2, 1e-1, "scale"]
                
                for gamma in gammas:
                
                    model = svc(C=C, max_iter=200, kernel=kernel, gamma=gamma)
                    
                    logscores = []
                    accs = []
                    
                    for j, ffold in enumerate(folds2):
                        ttrain, val = get_train_test_folds(df, ffold, j)
                        
                        X_train, y_train = ttrain.drop(columns=[TARGET]), ttrain[TARGET]
                        X_test, y_test = val.drop(columns=[TARGET]), val[TARGET]
                        
                        X_train, X_test = scale_data(X_train, X_test)
                        
                        model.fit(X_train, y_train)
                        
                        preds = model.predict(X_test)
                        pred_proba = model.predict_proba(X_test)
                        
                        logscore = log_score(y_test, pred_proba)
                        acc = accuracy(y_test, preds)
                        
                        logscores.append(logscore)
                        accs.append(acc)
                        
                        print(f"C: {C}, kernel: {kernel} gamma: {gamma} :: log score: {logscore}, accuracy: {acc}")
                    
                    mean_log = np.mean(logscores)
                    mean_acc = np.mean(accs)
                    
                    if mean_log > best_log:
                        best_log = mean_log
                        best_kernel = kernel
                        best_gamma = gamma
                        best_c = C
        # End inner CV
        
        # Fit on k-1 folds with best overall parameters
        X_train, y_train = train.drop(columns=[TARGET]), train[TARGET]
        X_test, y_test = test.drop(columns=[TARGET]), test[TARGET]
        
        X_train, X_test = scale_data(X_train, X_test)
        model = svc(C=best_c, max_iter=200, kernel=best_kernel, gamma=best_gamma)
        
        model.fit(X_train, y_train)
        
        # Evaluate on k_i
        preds = model.predict(X_test)
        pred_proba = model.predict_proba(X_test)
        
        logscore = log_score(y_test, pred_proba)
        acc = accuracy(y_test, preds)
        print(f"Best C: {C}, Best kernel: {kernel} :: log score: {logscore}, accuracy: {acc} (fold {i})")
        print()
        scores.append((logscore, acc))
        
        # Compute final bootstrapped confidence intervals
    final_log, (log_lb, log_ub) = bootstrap_metric([s[0] for s in scores])
    final_acc, (acc_lb, acc_ub) = bootstrap_metric([s[1] for s in scores])
    print(f"\nFinal Nested CV Log Score: {final_log:.4f} (CI: [{log_lb:.4f}, {log_ub:.4f}])")
    print(f"Final Nested CV Accuracy: {final_acc:.4f} (CI: [{acc_lb:.4f}, {acc_ub:.4f}])")
    
    
    return scores                 
            
def bootstrap_metric(metric_values, n_boot=1000, ci=95):
    """Bootstrap confidence intervals for a given metric."""
    boot_samples = np.random.choice(metric_values, (n_boot, len(metric_values)), replace=True)
    boot_means = np.mean(boot_samples, axis=1)
    lower_bound = np.percentile(boot_means, (100 - ci) / 2)
    upper_bound = np.percentile(boot_means, 100 - (100 - ci) / 2)
    return np.mean(boot_means), np.std(boot_means, ddof=1), (lower_bound, upper_bound)
        
def pprint(score, uncertainty):
    print(f"{score:.4f} +- {uncertainty:.4f}")
        
def compute_score_statistics(scores):
    final_log, se_log, (log_lb, log_ub) = bootstrap_metric([s[0] for s in scores])
    final_acc, se_acc, (acc_lb, acc_ub) = bootstrap_metric([s[1] for s in scores])
    print(f"\nFinal CV Log Score: {final_log:.4f} (CI: [{log_lb:.4f}, {log_ub:.4f}])")
    print(f"Final CV Accuracy: {final_acc:.4f} (CI: [{acc_lb:.4f}, {acc_ub:.4f}])")
    
    
    print("Log score:")
    pprint(final_log, se_log)
    print("Accuracy:")
    pprint(final_acc, se_acc)
    print()

    
    return (final_log, se_log), (final_acc, se_acc)
    
def encode_labels(df, columns):
    le = LabelEncoder()
    for col in columns:
        df[col] = le.fit_transform(df[col])    
        print(le.classes_)
    return df, le

def onehot_encode_feats(df):
    oh = OneHotEncoder(drop=None, sparse_output=False)
    encoded_cols = oh.fit_transform(df[categorical_cols])
    
    y_df = df["ShotType"]
    
    encoded_df = pd.DataFrame(encoded_cols, columns=oh.get_feature_names_out(categorical_cols))
    df_numeric = df[["Angle", "Distance", "Transition", "TwoLegged"]]
    
    df_final = pd.concat([encoded_df, df_numeric, y_df.to_frame()], axis=1)
    
    return df_final

def scale_data(X_train, X_test):
    return X_train, X_test
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test

if __name__ == "__main__":
    
    df = load_data()
    
    df, le = encode_labels(df, [TARGET] + categorical_cols)
    # df = onehot_encode_feats(df)
    
    _df, folds = get_stratified_folds(df, TARGET, k=5, shuffle=True)
    
    # Initialize models
    baseline_clf = baseline()
    
    logistic_clf = logistic_regression()
    
    svm = svc(C=.1, max_iter=200, kernel="rbf", gamma=.1)
    
    # Baseline scores
    print("Baseline:")
    baseline_scores, _ = cross_validate(_df, folds, baseline_clf)
    (blogs, blogs_se), (baccs, baccs_se) = compute_score_statistics(baseline_scores)
    print(baseline_scores)
    
    # LR scores
    print("Logistic regression:")
    lr_scores, _ = cross_validate(_df, folds, logistic_clf)
    (lrlogs, lrlogs_se), (lraccs, lraccs_se) = compute_score_statistics(lr_scores)
    print(lr_scores)
    
    start = time.time()
    # SVM scores
    print("SVM cross validation:")
    svm_scores, p2 = cross_validate(_df, folds, svm, True)

    errs, distances, comps = p2
    
    # Create separate columns for each fold (errors_0, errors_1, etc.)
    (svmlogs, svlogs_se), (svmaccs, svmaccs_se) = compute_score_statistics(svm_scores)
    
    # SVM training fold performance optimization
    print("SVM train fold optimization:")
    svm_train_scores = cross_validate_train_optimization(_df, folds)
    print(svm_train_scores)
    a, b = svm_train_scores[-2], svm_train_scores[-1]
    print(compute_score_statistics(zip(a,b)))
    
    # Create the DataFrame
    df = pd.DataFrame()
    
    folds = len(errs)
    for i in range(folds):
        
        errs[i] = [float(e) for e in errs[i]]
        _df = pd.DataFrame({f"log_score_{i}": errs[i], f"distance_{i}": distances[i], f"comp_{i}": comps[i]})
        df = pd.concat([df, _df], axis=1)

    df.to_csv("log_distance_errors.csv")
    end = time.time()
    
    print("SVM CV took", end-start)
    print("SVM iterations:", svm.n_iter_)
    
    print("SVM Nested CV")
    svm_nested_cv = nested_cross_validation(_df, folds)
    print(svm_nested_cv)
    
    
