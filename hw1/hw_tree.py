import csv
import numpy as np
import random
from collections import Counter
import time
import matplotlib.pyplot as plt

def all_columns(X: np.ndarray, rand: int):
    return range(X.shape[1])


def random_sqrt_columns(X: np.ndarray, rand: random.Random):
    n_cols = int(np.sqrt(X.shape[1]))
    cols = rand.sample(range(X.shape[1]), n_cols)
    return cols


class Tree:

    def __init__(
        self,
        rand: random.Random = None,
        get_candidate_columns=all_columns,
        min_samples: int = 2,
    ):
        self.rand = rand  # for replicability
        self.get_candidate_columns = get_candidate_columns  # needed for random forests
        self.min_samples = min_samples

    def _gini(self, y: np.ndarray):
        """Computes _gini impurity"""

        if len(y) == 0:
            return 0.0

        # TODO: Cache the value counts - count one side and avoid recounting the values!
        counts = Counter(y)
        probs = [count / len(y) for count in counts.values()]

        return 1 - sum(p**2 for p in probs)

    def _gini_for_split(
        self, X: np.ndarray, y: np.ndarray, feature, split_point: float
    ):
        """Returns _gini impurity for a suggested split"""

        indices_l = X[:, feature] <= split_point
        indices_r = X[:, feature] > split_point

        # Get labels from left and right split
        y_l = y[indices_l]
        y_r = y[indices_r]

        # Get _gini values
        _gini_l = self._gini(y_l)
        _gini_r = self._gini(y_r)

        # Get weights
        weight_l = len(y_l) / len(y)
        weight_r = len(y_r) / len(y)

        # Return total cost
        return (weight_l * _gini_l) + (weight_r * _gini_r), indices_l, indices_r

    def _get_split_points(self, feature: np.ndarray):
        """Returns array of values in between each pair of unique sorted feature values."""
        unique_values = np.unique(feature)

        if len(unique_values) < 2:
            return np.array([])  # No valid split points if only one unique value

        return (
            unique_values[:-1] + unique_values[1:]
        ) / 2  

    def _majority_class(self, y):
        values, counts = np.unique(y, return_counts=True)
        return values[np.argmax(counts)]

    def build(self, X: np.ndarray, y: np.ndarray):

        if len(X) < self.min_samples:
            prediction = self._majority_class(y)
            return TreeModel(
                best_feature=None,
                best_split=None,
                left_node=None,
                right_node=None,
                prediction=prediction
            )

        # TODO: We should take random features 1 per split not 1 per tree! (Reason: if you pick a bad combination of features the tree will be bad - if we do it every split the probablity for that is lower)
        features = self.get_candidate_columns(X, self.rand)  # A list of feature indices

        # Find best split
        # TODO: Optimize finding the best split value
        #     - Don't recount the values when looking for split points
        
        # Initialize
        best_cost = np.inf
        best_feature = None
        best_split = None
        best_left = None
        best_right = None

        for feature in features:
            for split_value in self._get_split_points(X[:, feature]):
                cost, indices_l, indices_r = self._gini_for_split(
                    X, y, feature, split_value
                )
                if cost < best_cost:
                    best_cost = cost
                    best_feature = feature
                    best_split = split_value
                    best_left = indices_l
                    best_right = indices_r

                if cost == 0:
                    break
                

        if best_feature is None:
            return TreeModel(
                best_feature=None,
                best_split=None,
                left_node=None,
                right_node=None,
                prediction=self._majority_class(y)
            )
            

        # Get the right data splits
        X_left, X_right = X[best_left, :], X[best_right, :]
        y_left, y_right = y[best_left], y[best_right]

        # Build left and right subtrees
        node_left = self.build(X_left, y_left)
        node_right = self.build(X_right, y_right)

        prediction = self._majority_class(y)

        return TreeModel(
            best_feature=best_feature,
            best_split=best_split,
            left_node=node_left,
            right_node=node_right,
            prediction=prediction
        )  # return an object that can do prediction


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
        mask_right = ~mask_left 

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
        self.rftree = Tree(
            rand, get_candidate_columns=random_sqrt_columns, min_samples=2
        )  # initialize the tree properly (initialize 1, use to build all trees)
        self.trees = []
        self.oob_samples = []

    def build(self, X, y):
        # ESLII - page 607

        n_samples = X.shape[0]

        # For b = 1:B
        for _ in range(self.n):
            # Get bootstrap sample
            sample_indices = self.rand.choices(
                range(n_samples), k=n_samples
            )  # Uses replacement by default
            # Bootstrap: If we have enough samples, we can approach the DGP really well
            
            oob_indices = list(set(range(n_samples)) - set(sample_indices))
            
            X_sample = X[sample_indices, :]
            y_sample = y[sample_indices]

            tree = self.rftree.build(X_sample, y_sample)

            self.trees.append(tree)
            self.oob_samples.append(oob_indices)

        return RFModel(self.trees, self.oob_samples, X, y)  # return an object that can do prediction


class RFModel:

    def __init__(self, trees: list, oob_samples, X, y):
        self.trees = trees
        self.oob_samples = oob_samples # list of OOB rows for corresponding tree
        self.X = X
        self.y = y


    def predict(self, X):
        # Use majority vote over all trees in forest
        predictions = np.array([tree.predict(X) for tree in self.trees])
        
        majority_votes = np.apply_along_axis(
            lambda x: np.bincount(x).argmax(), axis=0, arr=predictions
        )

        return majority_votes


    def importance(self):
        """
        Compute permutation-based variable importance using out-of-bag (OOB) samples.
        Returns a list of importance scores for each feature.
        OOB -> USE TRAIN DATA
        """
        if len(self.X) == 0:
            return self.importance()

        feature_importances = np.zeros(self.X.shape[1])
        total_trees = len(self.trees)

        for tree, oob_indices in zip(self.trees, self.oob_samples):
            # Identify OOB samples for this tree
            if len(oob_indices) == 0:
                continue  # Skip if no OOB samples

            X_oob = self.X[oob_indices, :]
            y_oob = self.y[oob_indices]

            # Compute baseline accuracy using OOB samples
            baseline_accuracy = np.mean(tree.predict(X_oob) == y_oob)

            # Permute each feature and measure accuracy drop
            n_feats = self.X.shape[1]
            for feature_idx in range(n_feats):
                # Permute in place
                X_oob[:, feature_idx] = np.random.permutation(X_oob[:, feature_idx])

                permuted_accuracy = np.mean(tree.predict(X_oob) == y_oob)

                feature_importances[feature_idx] += np.max(baseline_accuracy - permuted_accuracy, 0)
            
                # Unpermute
                X_oob[:, feature_idx] = self.X[oob_indices, feature_idx]


        # Average importance scores across all trees
        feature_importances /= total_trees
        
        # Return normalized feature importances
        return (
            feature_importances / np.sum(feature_importances)
            if np.sum(feature_importances) > 0
            else feature_importances
        )



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


def compute_metrics(truths, preds, n):
    # Compute train metrics
    misclf_rate = np.mean(truths != preds)
    misclf_sd = np.sqrt((misclf_rate * (1 - misclf_rate)) / n)

    return misclf_rate, misclf_sd


def hw_tree_full(train, test):
    """Builds a random forest on train data and reports accuracy
    and standard error when using train and test data as tests.
    """

    # Reporting the uncertainty
    # Bootstrap the test data and report the mean and average errors
    
    tree = Tree(rand=random.Random(1), get_candidate_columns=all_columns, min_samples=2)

    clf = tree.build(train[0], train[1])

    # Get predictions
    preds_train = clf.predict(train[0])
    preds_test = clf.predict(test[0])
    
    misclf_rate_train, misclf_sd_train = compute_metrics(
        train[1], preds_train, len(train[1])
    )
    misclf_rate_test, misclf_sd_test = compute_metrics(
        test[1], preds_test, len(test[1])
    )


    return (misclf_rate_train, misclf_sd_train), (misclf_rate_test, misclf_sd_test)


def hw_randomforests(train, test):
    """Builds a random forest on train data and reports accuracy
    and standard error when using train and test data as tests.
    """
    rf = RandomForest(rand=random.Random(0), n=100)

    rf = rf.build(train[0], train[1])

    # Get predictions
    preds_train = rf.predict(train[0])
    preds_test = rf.predict(test[0])
    

    misclf_rate_train, misclf_sd_train = compute_metrics(
        train[1], preds_train, len(train[1])
    )
    misclf_rate_test, misclf_sd_test = compute_metrics(
        test[1], preds_test, len(test[1])
    )

    start = time.time()
    importances2 = rf.importance()
    end = time.time()
    
    print("importances took ", end-start)    
    
    np.save("imp2.npy", importances2)

    plt.show()

    return (misclf_rate_train, misclf_sd_train), (misclf_rate_test, misclf_sd_test)


if __name__ == "__main__":

    learn, test, header = tki()

    start = time.time()

    print("full", hw_tree_full(learn, test))

    tree_end = time.time()

    print("random forests", hw_randomforests(learn, test))

    end = time.time()

    print(f"Tree took {(tree_end - start):.2f} seconds.")
    print(f"RF took {(end - tree_end):.2f} seconds.")
