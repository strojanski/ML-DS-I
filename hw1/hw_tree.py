import csv
import numpy as np
import random
from collections import Counter
import time
import matplotlib.pyplot as plt
import itertools
import tqdm
import pickle

RANDOM_SEED = 0

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
        self.used_features = set()

    def reset_used_feats(self):
        self.used_features = set()

    def _gini(self, y: np.ndarray):
        """Computes _gini impurity"""

        if len(y) == 0:
            return 0.0

        # TODO: Cache the value counts - count one side and avoid recounting the values!
        # Or rely on numpy being fast enough
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

        return (unique_values[:-1] + unique_values[1:]) / 2

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
                prediction=prediction,
                used_features=self.used_features,
            )

        # We should take random features 1 per split not 1 per tree! (Reason: if you pick a bad combination of features the tree will be bad - if we do it every split the probablity for that is lower)
        features = self.get_candidate_columns(X, self.rand)  # A list of feature indices

        # Find best split
        # Optimize finding the best split value
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
                prediction=self._majority_class(y),
                used_features=self.used_features,
            )

        self.used_features.add(best_feature)

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
            prediction=None,
            used_features=self.used_features,
        )  # return an object that can do prediction


class TreeModel:

    def __init__(
        self, best_feature, best_split, left_node, right_node, prediction, used_features
    ):
        self.best_feature = best_feature
        self.best_split = best_split
        self.left_node = left_node
        self.right_node = right_node
        self.prediction = prediction
        self.used_features = used_features

    def predict(self, X):
        predictions = [self.predict_sample(x) for x in X]
        return np.array(predictions)

    def predict_sample(self, x):
        # Detect leaf - just in case, should never happen
        if self.prediction is not None:
            return self.prediction

        # Traverse the tree
        node = self
        while node.left_node is not None and node.right_node is not None:
            node = (
                node.left_node
                if x[node.best_feature] <= node.best_split
                else node.right_node
            )

        return node.prediction


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

            self.rftree.reset_used_feats()

        return RFModel(
            self.trees, self.oob_samples, X, y, self.rand
        )  # return an object that can do prediction


class RFModel:

    def __init__(self, trees: list, oob_samples, X, y, rand):
        self.trees = trees
        self.oob_samples = oob_samples  # list of OOB rows for corresponding tree
        self.X = X
        self.y = y
        self.rand = rand

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
            # print("Features to check:", len(tree.used_features), "/", n_feats)
            for feature_idx in range(n_feats):

                if feature_idx not in tree.used_features:
                    feature_importances[feature_idx] += 0
                    continue

                # Permute in place
                X_oob[:, feature_idx] = np.random.permutation(X_oob[:, feature_idx])

                permuted_accuracy = np.mean(tree.predict(X_oob) == y_oob)

                feature_importances[feature_idx] += (
                    baseline_accuracy - permuted_accuracy
                )

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

    def importance3(self):
        """
        Compute permutation-based variable importance using out-of-bag (OOB) samples for tuples of 3 variables.
        Returns a list of importance scores for each feature.
        """

        feature_importances = {}

        start = time.time()
        count = 0
        for tree, oob_indices in zip(self.trees, self.oob_samples):
            count += 1
            # Identify OOB samples for this tree
            if len(oob_indices) == 0:
                continue  # Skip if no OOB samples

            X_oob = self.X[oob_indices, :]
            y_oob = self.y[oob_indices]

            # Compute baseline accuracy using OOB samples
            baseline_accuracy = np.mean(tree.predict(X_oob) == y_oob)

            # Permute each feature and measure accuracy drop
            n_feats = self.X.shape[1]
            feature_indices = range(n_feats)
            tested = set()

            other_feats = [f for f in feature_indices if f not in tree.used_features]

            # Add them for safekeeping
            for comb in itertools.combinations(other_feats, r=3):
                feats = tuple(sorted(comb))

                if feats not in feature_importances.keys():
                    feature_importances[feats] = 0
                else:
                    feature_importances[feats] += 0

            combs = list(itertools.combinations(tree.used_features, r=3))

            print(len(combs), len(tree.used_features))

            for comb in tqdm.tqdm(combs):
                feats = tuple(sorted(comb))

                if feats in tested:
                    continue
                else:
                    tested.add(feats)

                if feats not in feature_importances.keys():
                    feature_importances[feats] = 0

                # Permute in place
                for f in feats:
                    X_oob[:, f] = np.random.permutation(X_oob[:, f])

                permuted_accuracy = np.mean(tree.predict(X_oob) == y_oob)

                feature_importances[feats] += baseline_accuracy - permuted_accuracy

                # Unpermute
                for f in feats:
                    X_oob[:, f] = self.X[oob_indices, f]
            end = time.time()

            print("Tree took:", end - start, "seconds (", count, ")")

        with open("feature_imps3.pkl", "wb") as f:
            pickle.dump(feature_importances, f)

        tested = sorted(list(feature_importances.keys()))
        scores = []
        for t in tested:
            scores.append(feature_importances[t])

        return scores, tested

    def _get_feature_imps(self, node, imps, depth, curr_depth):
        if node.left_node is not None and node.right_node is not None:
            imps[node.best_feature] += 1
            depth[node.best_feature] += curr_depth

            if node.prediction is not None:
                return imps

            self._get_feature_imps(node.left_node, imps, depth, curr_depth+1)
            self._get_feature_imps(node.right_node, imps, depth, curr_depth+1)

        return imps, depth
    
            

    def importance3_structure(self, account_for_depth=False):
        """Traverses the tree with no known data and determines the feature importances based on the number of splits done with it"""
        feature_importances = np.zeros(self.X.shape[1])
        depth = np.zeros(self.X.shape[1])
        print("imp3 structure")
        for i, tree in enumerate(self.trees):
            
            node = tree
            feature_importances, depth = self._get_feature_imps(node, feature_importances, depth, 0) # Used features have nonzero values


        # normalize depth and feature_imps
        feature_importances /= sum(feature_importances)
        log_depth = 1 / np.log(depth + 2)
        
        # Now find all possible combinations of the actually used combinations and simply compute the most used combination
        all_features = np.nonzero(feature_importances)[0]
        all_combs = list(itertools.combinations(list(all_features), r=3))
        
        max_importance = 0
        best_comb = None
        for comb in all_combs:
            feats = tuple(sorted(comb))
            importance = 0
            for feat in feats:
                if account_for_depth:
                    importance += feature_importances[feat] + log_depth[feat]
                else:
                    importance += feature_importances[feat]
            if importance > max_importance:
                max_importance = importance
                best_comb = feats    

        return best_comb, max_importance


def get_imp3_columns(X, rand=None):
    # return [49, 162, 259]   # log scale 33.4
    # return[274, 279, 362]   # Inverse freq, normalized, 38
    return [274, 275, 361]  # importances3 best feats


def get_imp_columns(X, rand=None):
    # return [276, 330, 379]  # without depth
    return [275, 274, 276]


def compare_importances(train, test, rand):
    # importance 3 best feats: 274., 275., 361.
    tree_imp = Tree(rand, get_imp_columns, min_samples=2)
    tree_imp3 = Tree(rand, get_imp3_columns, min_samples=2)

    t1 = tree_imp.build(train[0], train[1])
    t3 = tree_imp3.build(train[0], train[1])

    preds_imp1_test = t1.predict(test[0])
    preds_imp3_test = t3.predict(test[0])

    preds_imp1_train = t1.predict(train[0])
    preds_imp3_train = t3.predict(train[0])

    n_boot = 200
    imp1_trains, imp1_tests = bootstrap_error(
        train, test, preds_imp1_train, preds_imp1_test, n_boot, rand
    )
    imp3_trains, imp3_tests = bootstrap_error(
        train, test, preds_imp3_train, preds_imp3_test, n_boot, rand
    )

    misclf_rate_test_imp1 = np.mean(imp1_tests)
    misclf_sd_test_imp1 = np.std(imp1_tests, ddof=1) / np.sqrt(n_boot)

    misclf_rate_test_imp3 = np.mean(imp3_tests)
    misclf_sd_test_imp3 = np.std(imp3_tests, ddof=1) / np.sqrt(n_boot)

    print(
        f"Misclassification rate on features from 1000 trees, test set, single importances: {misclf_rate_test_imp1:.4f} +- {misclf_sd_test_imp1:.4f}"
    )
    print(
        f"Misclassification rate on features from 1000 trees, test set, 3 importances: {misclf_rate_test_imp3:.4f} +- {misclf_sd_test_imp3:.4f}"
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

    # Bootstrap errors:
    n_boot = 200
    trains, tests = bootstrap_error(train, test, preds_train, preds_test, n_boot)

    misclf_rate_train = np.mean(trains)
    misclf_sd_train = np.std(trains, ddof=1) / np.sqrt(n_boot)

    misclf_rate_test = np.mean(tests)
    misclf_sd_test = np.std(tests, ddof=1) / np.sqrt(n_boot)

    return (misclf_rate_train, misclf_sd_train), (misclf_rate_test, misclf_sd_test)


def get_nonrandom_imps(train):

    tree = Tree(None, get_candidate_columns=all_columns, min_samples=len(train[0]) - 1)
    root_feats = []
    for i in range(100):
        print(i)

        rand_samples = np.random.choice(
            range(len(train[0])), len(train[0]), replace=True
        )

        X = train[0][rand_samples, :]
        y = train[1][rand_samples]

        tm = tree.build(X, y)

        root_feats.append(tm.best_feature)

    return root_feats


def bootstrap_error(train, test, preds_train, preds_test, n_boot=100, rand=random.Random(RANDOM_SEED)):
    trains, tests = [], []
    for i in range(n_boot):
        ix_boot_preds_test = rand.choices(
            range(len(preds_test)), k=len(preds_test))
        ix_boot_preds_train = rand.choices(
            range(len(preds_test)), k=len(preds_train))

        misclf_rate_train, misclf_sd_train = compute_metrics(
            train[1][ix_boot_preds_train],
            preds_train[ix_boot_preds_train],
            len(train[1]),
        )

        misclf_rate_test, misclf_sd_test = compute_metrics(
            test[1][ix_boot_preds_test], preds_test[ix_boot_preds_test], len(test[1])
        )

        trains.append(misclf_rate_train)
        tests.append(misclf_rate_test)
        # trains.append((misclf_rate_train, misclf_sd_train))
        # tests.append((misclf_rate_test, misclf_sd_test))

    return trains, tests


def hw_randomforests(train, test, plot=False):
    """Builds a random forest on train data and reports accuracy
    and standard error when using train and test data as tests.
    """
    rf = RandomForest(rand=random.Random(RANDOM_SEED), n=100)

    rf = rf.build(train[0], train[1])

    # Get predictions
    preds_train = rf.predict(train[0])
    preds_test = rf.predict(test[0])

    # Bootstrap errors:
    n_boot = 200
    trains, tests = bootstrap_error(train, test, preds_train, preds_test, n_boot)

    misclf_rate_train = np.mean(trains)
    misclf_sd_train = np.std(trains, ddof=1) / np.sqrt(n_boot)

    misclf_rate_test = np.mean(tests)
    misclf_sd_test = np.std(tests, ddof=1)  / np.sqrt(n_boot)

    start = time.time()
    importances = rf.importance()
    end = time.time()

    np.save("importances_normal.npy", importances)
    # np.savetxt("importances_normal_names.txt", names)

    # Plot importances and non-random tree importances
    if plot:
        nonrandom_feats = get_nonrandom_imps(train)

        nonrandom_imps = np.zeros(importances.shape)
        nonrandom_imps[nonrandom_feats] = importances[nonrandom_feats]

        plt.figure(figsize=(9, 6))
        plt.bar(range(len(importances)), importances, label="RF importances")
        plt.bar(range(len(importances)), nonrandom_imps, label="Root importances")
        plt.legend()
        plt.show()

    print("importances took ", end - start)

    plt.show()

    return (misclf_rate_train, misclf_sd_train), (misclf_rate_test, misclf_sd_test)


def hw_randomforests_sized(train, test, plot=False):
    """Builds a random forest and tests forests with different amount of trees - reports test error
    """
    
    results = []
    
    for n in range(1, 502, 50):

        print(n)
        
        rf = RandomForest(rand=random.Random(0), n=n)

        rf = rf.build(train[0], train[1])

        # Get predictions
        preds_train = rf.predict(train[0])
        preds_test = rf.predict(test[0])

        # Bootstrap errors:
        n_boot = 200
        trains, tests = bootstrap_error(train, test, preds_train, preds_test, n_boot)

        # misclf_rate_train = np.mean(trains)
        # misclf_sd_train = np.std(trains, ddof=1) / np.sqrt(n_boot)

        misclf_rate_test = np.mean(tests)
        misclf_sd_test = np.std(tests, ddof=1) / np.sqrt(n_boot)

        results.append((n, misclf_rate_test, misclf_sd_test))
   
    return results


def test_rf_importance_unknown(train):
    rf = RandomForest(random.Random(RANDOM_SEED), n=100)
    rf = rf.build(train[0], train[1])

    best_comb, max_imp = rf.importance3_structure(account_for_depth=False)

    print("importances3_structure: best features and common number of splits on 100 trees:")
    print(best_comb, max_imp)
    print("--------------------------")


if __name__ == "__main__":

    learn, test, header = tki()

    start = time.time()

    # 1.1
    print("full", hw_tree_full(learn, test))

    tree_end = time.time()

    # Full random forest - return misclf rate & compute importances (1.2 + 2 (importances))
    print("random forests", hw_randomforests(learn, test))

    end = time.time()

    print(f"Tree took {(tree_end - start):.2f} seconds.")
    print(f"RF took {(end - tree_end):.2f} seconds.")
    
    # Get misclf rate based on number of trees (1.3)
    # results = hw_randomforests_sized(learn, test)
    # results = np.array(results)
    # np.save("results.npy", results)

    # Compare importances with importances 3 (tree with top 3 features), 3.1
    compare_importances(learn, test, random.Random(RANDOM_SEED))
    
    # 3.2 (imp3_structure)
    test_rf_importance_unknown(learn)
    
    # 
