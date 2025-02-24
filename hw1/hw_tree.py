import csv
import numpy as np
import random
from collections import Counter
from sklearn.metrics import accuracy_score

def all_columns(X: np.ndarray, rand: int):
    return range(X.shape[1])


def random_sqrt_columns(X: np.ndarray, rand: random.Random):
    n_cols = int(np.sqrt(X.shape[1]))  
    cols = rand.sample(range(X.shape[1]), n_cols)  
    return cols


class Tree:
    
    def __init__(self, rand: random.Random = None,
                 get_candidate_columns = all_columns,
                 min_samples: int = 2):
        self.rand = rand  # for replicability
        self.get_candidate_columns = get_candidate_columns  # needed for random forests
        self.min_samples = min_samples

    def gini(self, y: np.ndarray):
        """Computes gini impurity"""
        
        if len(y) == 0:
            return 0.0
        
        counts = Counter(y)
        probs = [count / len(y) for count in counts.values()]
        
        return 1 - sum(p ** 2 for p in probs)
    
    def gini_for_split(self, X: np.ndarray, y: np.ndarray, feature, split_point: float):
        """Returns gini impurity for a suggested split"""
        
        indices_l = X[:, feature] <= split_point
        indices_r = X[:, feature] > split_point
        
        # Get labels from left and right split
        y_l = y[indices_l]
        y_r = y[indices_r]
        
        # Get gini values
        gini_l = self.gini(y_l)
        gini_r = self.gini(y_r)
        
        # Get weights
        weight_l = len(y_l) / len(y)
        weight_r = len(y_r) / len(y)
        
        # Return total cost
        return (weight_l * gini_l) + (weight_r * gini_r), indices_l, indices_r
        
    
    def get_split_points(self, feature: np.ndarray):
        """Returns array of values in between each pair of unique sorted feature values."""
        unique_values = np.unique(feature)
        
        if len(unique_values) < 2:
            return np.array([])  # No valid split points if only one unique value
        
        return (unique_values[:-1] + unique_values[1:]) / 2


    def majority_class(self, y):
        values, counts = np.unique(y, return_counts=True)
        return values[np.argmax(counts)]


    def build(self, X: np.ndarray, y: np.ndarray):
        
        if len(X) < self.min_samples:
            prediction = self.majority_class(y)
            return TreeModel(best_feature=None, best_split=None, left_node=None, right_node=None, prediction=prediction)
               
        
        features = self.get_candidate_columns(X, self.rand)  # A list of feature indices
        
        # Initialize
        best_cost = np.inf
        best_feature = None
        best_split = None
        best_left = None
        best_right = None

        # Find best split
        for feature in features:
            for split_value in self.get_split_points(X[:, feature]):
                cost, indices_l, indices_r = self.gini_for_split(X, y, feature, split_value)
                if cost < best_cost:
                    best_cost = cost
                    best_feature = feature
                    best_split = split_value
                    best_left = indices_l
                    best_right = indices_r
                
                if cost == 0:
                    break
                    
        if best_feature is None:
            return TreeModel(best_feature=None, best_split=None, left_node=None, right_node=None, prediction=self.majority_class(y))
                    
        # Get the right data splits
        X_left, X_right = X[best_left, :], X[best_right, :]
        y_left, y_right = y[best_left], y[best_right]

        # Build left and right subtrees
        node_left = self.build(X_left, y_left)
        node_right = self.build(X_right, y_right)
                    
        prediction = self.majority_class(y)
        
        return TreeModel(best_feature=best_feature,
                         best_split=best_split,
                         left_node=node_left,
                         right_node=node_right,
                         prediction=prediction)  # return an object that can do prediction


class TreeModel:

    def __init__(self, best_feature, best_split, left_node, right_node, prediction):
        self.best_feature = best_feature
        self.best_split = best_split
        self.left_node = left_node
        self.right_node = right_node
        self.prediction = prediction
        
    def predict(self, X):
        """Recursively predicts labels for given X"""
        # Detect leaf
        if self.best_feature is None:
            return np.full(len(X), self.prediction)
    
        # Otherwise split X 
        mask_left = X[:, self.best_feature] <= self.best_split
        mask_right = ~mask_left  # Avoid redundant comparison
        
        # Recursively get predictions
        preds_left = self.left_node.predict(X[mask_left])
        preds_right = self.right_node.predict(X[mask_right])

        # Combine preds
        predictions = np.empty(len(X), dtype=int)
        predictions[mask_left] = preds_left
        predictions[mask_right] = preds_right

        return predictions


class RandomForest:

    def __init__(self, rand=None, n=50):
        self.n = n
        self.rand = rand
        self.rftree = Tree(...)  # initialize the tree properly

    def build(self, X, y):
        # ...
        return RFModel(...)  # return an object that can do prediction


class RFModel:

    def __init__(self):
        pass
        # ...

    def predict(self, X):
        # ...
        return None

    def importance(self):
        imps = np.zeros(self.X.shape[1])
        # ...
        return imps


def read_tab(fn, adict):
    content = list(csv.reader(open(fn, "rt"), delimiter="\t"))

    legend = content[0][1:]
    data = content[1:]

    X = np.array([d[1:] for d in data], dtype=float)
    y = np.array([adict[d[0]] for d in data])

    return legend, X, y


def tki():
    legend, Xt, yt = read_tab("tki-train.tab", {"Bcr-abl": 1, "Wild type": 0})
    _, Xv, yv = read_tab("tki-test.tab", {"Bcr-abl": 1, "Wild type": 0})
    return (Xt, yt), (Xv, yv), legend


def hw_tree_full(train, test):
    tree = Tree(rand=random.Random(1),
                        get_candidate_columns=all_columns,
                        min_samples=2)
    
    clf = tree.build(train[0], train[1])
    
    preds = clf.predict(test[0])
    
    print(accuracy_score(test[1], preds))
    
    return preds
    
if __name__ == "__main__":
    
    learn, test, legend = tki()
    
    hw_randomforests = None

    print("full", hw_tree_full(learn, test))
    print("random forests", hw_randomforests(learn, test))
