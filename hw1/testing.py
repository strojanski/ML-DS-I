import unittest
import numpy as np
from collections import Counter
from hw_tree import Tree, TreeModel


class MyTests(unittest.TestCase):
    
    def setUp(self):
        """Initialize a Tree instance before each test."""
        self.tree = Tree()
    
    def test__gini(self):
        """Test _gini impurity calculation."""
        self.assertAlmostEqual(self.tree._gini(np.array(["A", "A", "B", "B"])), 0.5)
        self.assertAlmostEqual(self.tree._gini(np.array(["A", "A", "A", "A"])), 0.0)
        self.assertAlmostEqual(self.tree._gini(np.array(["A", "B", "C", "D"])), 0.75)
        self.assertAlmostEqual(self.tree._gini(np.array([])), 0.0)

    def test__get_split_points(self):
        """Test that split points are generated correctly."""
        feature = np.array([1, 2, 3, 5, 7])
        expected = np.array([1.5, 2.5, 4.0, 6.0])  # Midpoints
        np.testing.assert_array_almost_equal(self.tree._get_split_points(feature), expected)

        # Single unique value should return an empty array
        feature = np.array([5, 5, 5])
        self.assertEqual(len(self.tree._get_split_points(feature)), 0)

    def test___gini_for_split(self):
        """Test _gini calculation for a split."""
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array(["A", "A", "B", "B", "B"])
        
        cost, left_indices, right_indices = self.tree.__gini_for_split(X, y, feature=0, split_point=2.5)

        self.assertLess(cost, 1.0)  # _gini should always be between 0 and 1
        self.assertTrue(np.all(left_indices == np.array([True, True, False, False, False])))
        self.assertTrue(np.all(right_indices == np.array([False, False, True, True, True])))

    def test__majority_class(self):
        """Test that the majority class is correctly determined."""
        y = np.array(["A", "A", "B", "B", "B"])
        self.assertEqual(self.tree._majority_class(y), "B")
        
        y = np.array(["A", "A", "B", "B"])
        self.assertIn(self.tree._majority_class(y), ["A", "B"])  # Both A and B are tied

    def test_tree_build_and_predict(self):
        """Test tree building and prediction."""
        X = np.array([[0, 0],
                      [0, 1],
                      [1, 0],
                      [1, 1]])
        y = np.array([0, 0, 1, 1])

        t = Tree(min_samples=1)  # Ensure it doesn't stop early
        model = t.build(X, y)

        predictions = model.predict(X)
        np.testing.assert_array_equal(predictions, y)

    def test_tree_handles_small_dataset(self):
        """Test tree behavior when the dataset is too small to split."""
        X = np.array([[0, 0]])
        y = np.array([1])

        t = Tree(min_samples=2)  # Force early stopping
        model = t.build(X, y)

        self.assertIsNone(model.best_feature)
        self.assertEqual(model.prediction, 1)
        np.testing.assert_array_equal(model.predict(X), y)

    def test_tree_build_and_predict_multilevel(self):
        """Test tree building and prediction on a dataset requiring multiple splits."""
        X = np.array([[2, 3],  # Class 0
                      [2, 2],  # Class 0
                      [8, 3],  # Class 1
                      [8, 2],  # Class 1
                      [5, 5],  # Class 2
                      [6, 6]]) # Class 2
        y = np.array([0, 0, 1, 1, 2, 2])

        t = Tree(min_samples=1)
        model = t.build(X, y)

        predictions = model.predict(X)
        np.testing.assert_array_equal(predictions, y)

    def test_tree_generalization(self):
        """Test tree predictions on unseen data after training on a complex dataset."""
        X_train = np.array([[1, 2], [2, 3], [3, 4], [5, 6], [6, 7], [7, 8]])
        y_train = np.array([0, 0, 0, 1, 1, 1])

        X_test = np.array([[4, 5], [8, 9], [1, 1], [6, 6]])
        y_expected = np.array([0, 1, 0, 1])  # Expected based on nearest neighbors

        t = Tree(min_samples=1)
        model = t.build(X_train, y_train)

        predictions = model.predict(X_test)
        np.testing.assert_array_equal(predictions, y_expected)

    def test_tree_train_test_split(self):
        """Test tree predictions on a separate test set after training."""
        X_train = np.array([[1, 2], [2, 3], [3, 4], [5, 6], [6, 7], [7, 8]])
        y_train = np.array([0, 0, 0, 1, 1, 1])

        X_test = np.array([[4, 5], [8, 9], [1, 1], [6, 6]])
        y_test = np.array([0, 1, 0, 1])  # Expected predictions

        t = Tree(min_samples=1)
        model = t.build(X_train, y_train)

        predictions = model.predict(X_test)
        np.testing.assert_array_equal(predictions, y_test)

if __name__ == '__main__':
    unittest.main()
