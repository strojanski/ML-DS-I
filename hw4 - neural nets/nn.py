import numpy as np
import csv

class Layer:
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim  # Input dimension = # of features (from prev layer)
        self.output_dim = output_dim # Output dimension = # of neurons
        self.weights = np.random.randn(input_dim, output_dim) * np.sqrt(1. / (input_dim)) * 2   # Xavier
        self.biases = np.zeros((1, output_dim))

    def forward(self, X):
        self.X = X
        self.output = np.dot(X, self.weights) + self.biases
        return self.output
    
    def backward(self, dZ):
        self.dweights = np.dot(self.X.T, dZ)
        self.dbiases = np.sum(dZ, axis=0, keepdims=True)
        self.dX = np.dot(dZ, self.weights.T)
        return self.dX
    
class Softmax:
    """Softmax activation function, used in the last layer - outputs probabilities"""
    def forward(self, Z):
        self.Z = Z
        
        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        probs = exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
        
        self.probs = probs
        return probs
    
    # def backward(self, dZ):
    #     # return dZ
    #     self.dinp = np.zeros_like(dZ)
        
    #     for i, (prob, dz) in enumerate(zip(self.probs, dZ)):
    #         prob = prob.reshape(-1, 1) 
            
    #         # Jacobian
    #         jacobian = np.diagflat(prob) - np.dot(prob, prob.T)
    #         self.dinp[i] = np.dot(jacobian, dz)
    #     return self.dinp

class Sigmoid_Activation:
    """Sigmoid activation function, used in between hidden layers"""
    def __init__(self):
        self.Z = None
        self.output = None
        self.dinput = None
        
    def forward(self, Z):
        self.Z = Z
        self.output = 1 / (1 + np.exp(-1 * Z))
        return self.output
    
    def backward(self, dZ):
        self.dinput = dZ * self.output * (1 - self.output)
        return self.dinput
    
class CategoricalCrossEntropyLoss:
    """Categorical Cross-Entropy loss function, as the loss function (global)"""
    def forward(self, y_pred, y_true):
        """y_pred = softmax output, y_true = labels, need to one-hot encode"""
        n_samples = len(y_true)
        
        y_pred = np.clip(y_pred, 1e-8, 1 - 1e-8)  # Avoid log(0)
        
        confidences = y_pred[range(n_samples), y_true]
        
        # Onehot encoded
        if len(y_true.shape) == 2:
            confidences = np.sum(y_pred * y_true, axis=1)
            
        neg_log_likelihood = -np.log(confidences)
        return np.mean(neg_log_likelihood)
    
    # def backward(self, dy, y_true):
    #     n_samples = len(dy)
    #     labels = len(dy[0])
        
    #     # Onehot
    #     if len(y_true.shape) == 1:
    #         y_true = np.eye(labels)[y_true]
        
    #     self.dinput = -y_true / dy
    #     self.dinput /= n_samples    # Normalize gradient
    #     return self.dinput   

class Softmax_CrossEntropyLoss:
    def __init__(self):
        self.activation = Softmax()
        self.loss = CategoricalCrossEntropyLoss()
        
    def forward(self, logits, y_true):
        self.probs = self.activation.forward(logits)
        return self.probs, self.loss.forward(self.probs, y_true)
    
    def backward(self, y_true):
        n_samples = len(self.probs)
        
        # In case we one-hot encode labels
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        
        self.dinputs = self.probs.copy()
        self.dinputs[range(n_samples), y_true] -= 1
        self.dinputs /= n_samples
        
        return self.dinputs
        

class SGD_Optimizer:
    def __init__(self, lr=0.01):
        self.lr = lr
        
    def update(self, layer):
        layer.weights -= self.lr * layer.dweights
        layer.biases -= self.lr * layer.dbiases
        
        return layer

class ANNRegression:
    # implement me too, please
    pass


# data reading

def read_tab(fn, adict):
    content = list(csv.reader(open(fn, "rt"), delimiter="\t"))

    legend = content[0][1:]
    data = content[1:]

    X = np.array([d[1:] for d in data], dtype=float)
    y = np.array([adict[d[0]] for d in data])

    return legend, X, y


def doughnut():
    legend, X, y = read_tab("doughnut.tab", {"C1": 0, "C2": 1})
    return X, y


def squares():
    legend, X, y = read_tab("squares.tab", {"C1": 0, "C2": 1})
    return X, y


class ANNClassification:
    def __init__(self, units, lambda_=0, n_iter=10000, verbose=False):
        self.units = units
        self.softmax = Softmax()
        self.loss = CategoricalCrossEntropyLoss()
        self.softmax_cce = Softmax_CrossEntropyLoss()
        self.optimizer = SGD_Optimizer(lr=1)
        self.layers = []
        self.n_iter = n_iter
        self.lambda_ = lambda_
        self.verbose = verbose
        
    def fit(self, X, y):
        self.build(X, y)
        
        for i in range(self.n_iter):
            # Decay
            self.lr = 1 * 0.99**i
            
            logits = self.forward_pass(X)
            probs, loss = self.softmax_cce.forward(logits, y)
            
            preds = np.argmax(probs, axis=1)
            acc = np.mean(preds == y)
            if i % 100 == 0: 
                if acc == 1 and loss < 0.01:
                    print("Converged after ", i, "iterations")
                    break
                
                if self.verbose:
                    print(f"Accuracy: {acc:.4f}")
                    print(f"Loss: {loss:.4f}")
                    
            
            self.backward_pass(X, y)
            
        print(f"Accuracy: {acc:.4f}")
        print(f"Loss: {loss:.4f}")
            
        return self
        
    def build(self, X, y):
        self.units = [X.shape[1]] + self.units + [len(set(y))]
        print(self.units)

        for i in range(len(self.units) - 1):
            self.layers.append(Layer(self.units[i], self.units[i + 1]))

        self.activations = [Sigmoid_Activation() for _ in range(len(self.units) - 2)]

        
    def forward_pass(self, X):
        inp = X
        for i, layer in enumerate(self.layers[:-1]):
            Z = layer.forward(inp)
            inp = self.activations[i].forward(Z)

        logits = self.layers[-1].forward(inp)

        return logits
    
    def backward_pass(self, X, y):
        # Softmax + Loss 
        dL = self.softmax_cce.backward(y)
        
        # Last layer
        d = self.layers[-1].backward(dL)

        # Other layers
        for act, layer in zip((reversed(self.activations)), reversed(self.layers[:-1])):
            d = act.backward(d)
            d = layer.backward(d)
        for layer in self.layers:
            self.optimizer.update(layer)
        
    
    def predict(self, X):
        logits = self.forward_pass(X)
        probs = self.softmax.forward(logits)
        return probs
        # predictions = np.argmax(probs, axis=1)
        # return np.eye(len(set(predictions)))[predictions]    

from sklearn.model_selection import train_test_split
if __name__ == "__main__":

    fitter = ANNClassification(units=[10, 10], lambda_=0., verbose=True)
    X, y = squares()
    X, y = doughnut()
    
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    print(X_train.shape, y_train.shape)
    
    
    m = fitter.fit(X_train, y_train)
    print()
    pred = m.predict(X_test)
    
    pred = np.argmax(pred, axis=1)
    print("Accuracy on test set:", np.mean(pred == y_test))
    
    exit()
    
    
    X = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ], dtype=float)
    y = np.array([0, 1, 2])
    
    X,y = doughnut()
    print(X.shape, y.shape)
    
    model = TestNN(X=X, y=y)
    model.fit()
    
    exit()
    model = fitter.fit(X, y)
    predictions = fitter.predict(X)
    print(predictions)
    np.testing.assert_almost_equal(predictions,
                                   [[1, 0, 0],
                                    [0, 1, 0],
                                    [0, 0, 1]], decimal=3)
