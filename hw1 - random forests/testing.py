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

    def test_gini_for_split(self):
        """Test _gini calculation for a split."""
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array(["A", "A", "B", "B", "B"])
        
        cost, left_indices, right_indices = self.tree._gini_for_split(X, y, feature=0, split_point=2.5)

        self.assertLess(cost, 1.0)  # _gini should always be between 0 and 1
        self.assertTrue(np.all(left_indices == np.array([True, True, False, False, False])))
        self.assertTrue(np.all(right_indices == np.array([False, False, True, True, True])))



if __name__ == '__main__':
    unittest.main()
