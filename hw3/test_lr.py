import random
import unittest

import numpy as np

from solution import MultinomialLogReg, OrdinalLogReg
import pandas as pd

# class HW2Tests(unittest.TestCase):

#     def setUp(self):
#         self.X = np.array([[0, 0],
#                            [0, 1],
#                            [1, 0],
#                            [1, 1],
#                            [1, 1]])
#         self.y = np.array([0, 0, 1, 1, 2])
#         self.train = self.X[::2], self.y[::2]
#         self.test = self.X[1::2], self.y[1::2]

#     def test_multinomial(self):
#         l = MultinomialLogReg()
#         c = l.build(self.X, self.y)
#         prob = c.predict(self.test[0])
#         self.assertEqual(prob.shape, (2, 3))
#         self.assertTrue((prob <= 1).all())
#         self.assertTrue((prob >= 0).all())
#         np.testing.assert_almost_equal(prob.sum(axis=1), 1)

#     def test_ordinal(self):
#         l = OrdinalLogReg()
#         c = l.build(self.X, self.y)
#         prob = c.predict(self.test[0])
#         self.assertEqual(prob.shape, (2, 3))
#         self.assertTrue((prob <= 1).all())
#         self.assertTrue((prob >= 0).all())
#         np.testing.assert_almost_equal(prob.sum(axis=1), 1)

class MyTests(unittest.TestCase):
    
    def setUp(self):
        self.X1 = np.array([
            [0],[0],[0],[1],[1],[1]
        ]) 
        self.y1 = np.array([0,0,0,1,1,1])
        
        self.X2 = np.array([
            [2,3],
            [3,5],
            [5,1],
            [8, 17],
            [0,0],
            [7,7]]
            )
        self.y2 = np.array([0,0,0,0,0,0])
        
        self.X3 = np.array([
            [0,0],
            [0,1],
            [0,2],
            [0,5],
            [2,4],
            [2,2],
            [3,0],
            [3,0],
            [3,0],
        ])
        self.y3 = np.array([0,0,0,0,1,1,1,1,1])
        
        
    def test_simple(self):
        train = self.X1[::2], self.y1[::2]
        test = self.X1[1::2], self.y1[1::2]
        
        l = MultinomialLogReg()
        c = l.build(train[0], train[1])
        
        prob = c.predict(test[0])
        
        preds = np.argmax(prob, axis=1)
        
        self.assertTrue(all(p == r for p, r in zip(preds, test[1])))

    def test_single_class(self):
        train = self.X2[::2], self.y2[::2]
        test = self.X2[1::2], self.y2[1::2]
        
        l = MultinomialLogReg()
        c = l.build(train[0], train[1])
        
        prob = c.predict(test[0])
        
        preds = np.argmax(prob, axis=1)
        
        self.assertTrue(all(p == r for p, r in zip(preds, test[1])))

    def test_ordinal(self):
        train = self.X3[::2], self.y3[::2]
        test = self.X3[1::2], self.y3[1::2]
        
        l = OrdinalLogReg()
        l = MultinomialLogReg()
        c = l.build(train[0], train[1])
        
        prob = c.predict(test[0])
        preds = np.argmax(prob, axis=1)
        # print(preds)
        # print(test[1])
        
        # print(c.theta)
        # print(c.thresh)
        
        
    #     self.assertTrue(all(p == r for p,r in zip(preds, test[1])))

if __name__ == "__main__":
    unittest.main()


