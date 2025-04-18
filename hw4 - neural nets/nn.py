import numpy as np
import csv

class Layer:
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim  # Input dimension = # of features (from prev layer)
        self.output_dim = output_dim # Output dimension = # of neurons
        self.weights = np.random.randn(input_dim, output_dim) * np.sqrt(2. / (input_dim + output_dim))  # Xavier
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
    def forward(self, Z):
        self.Z = Z
        
        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        probs = exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
        
        self.probs = probs
        return probs
    
    def backward(self, dZ):
        # self.dinp = np.zeros_like(dZ)
        
        # for i, (prob, dz) in enumerate(zip(self.probs, dz)):
        #     prob = prob.reshape(-1, 1) 
            
        #     # Jacobian
        #     jacobian = np.diagflat(prob) - np.dot(prob, prob.T)
        #     self.dinp[i] = np.dot(jacobian, dz)
        # return self.dinp
        return dZ
    
class Sigmoid_Activation_CrossEntropy():
    def __init__(self):
        self.sigmoid = Sigmoid_Activation()
        self.loss = BinaryCrossEntropyLoss()  # Corrected to use BinaryCrossEntropyLoss
    
    def forward(self, inputs, y_true):
        output = self.sigmoid.forward(inputs)
        return self.loss.forward(output, y_true)
    
    def backward(self, dy, y_true):
        # Gradient for binary cross-entropy with sigmoid activation
        n_samples = len(dy)
        self.dinputs = dy.copy()
        
        self.dinputs = self.sigmoid.backward(self.dinputs)  # Use sigmoid's backward pass
        self.dinputs /= n_samples  # Normalize gradient
        return self.dinputs


class Sigmoid_Activation:
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
    def forward(self, y_pred, y_true):
        n_samples = len(y_true)
        
        y_pred = np.clip(y_pred, 1e-8, 1 - 1e-8)  # Avoid log(0)
        
        confidences = y_pred[range(n_samples), y_true]
        
        # Onehot encoded
        if len(y_true.shape) == 2:
            confidences = np.sum(y_pred * y_true, axis=1)
            
        neg_log_likelihood = -np.log(confidences)
        return np.mean(neg_log_likelihood)
    
    def backward(self, dy, y_true):
        n_samples = len(dy)
        labels = len(dy[0])
        
        # Onehot
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        
        self.dinput = -y_true / dy
        self.dinput /= n_samples    # Normalize gradient
        return self.dinput   

class ANNClassification:
    # implement me
    def __init__(self, units, lambda_):
        self.units = units
        self.lambda_ = lambda_
        self.model = None
        self.softmax = Softmax()
        self.activation = Sigmoid_Activation()
        self.loss = CategoricalCrossEntropyLoss()
        
    def sigmoid(self, X):
        return 1 / (1 + np.exp(-X))

    def fit(self, X, y):
        input_dim = X.shape[1]
        output_dim = len(np.unique(y)) 
        self.units = [input_dim] + self.units + [output_dim]
        self.build()

        
    def build(self):
        # Build the ANN model here
        self.layers = []
        for i in range(len(self.units) - 1):
           self.layers.append(Layer(self.units[i], self.units[i + 1]))
        self.activation = Sigmoid_Activation()
        self.loss = CategoricalCrossEntropyLoss()
            
    def forward_pass(self, X):
        # Forward pass through the network
        inp = X
        for layer in self.layers[:-1]:
            Z = layer.forward(inp)
            inp = self.activation.forward(Z)

        Z = self.layers[-1].forward(inp)
        Z = self.softmax.forward(Z)

        return Z

    def predict(self, X):
        # Make predictions using the trained model
        Z = self.forward_pass(X)
        predictions = np.argmax(Z, axis=1)
        return np.eye(len(set(predictions)))[predictions]

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


if __name__ == "__main__":

    # example NN use
    fitter = ANNClassification(units=[3,4], lambda_=0)
    sigmoid_outputs = np.array([[0.7, 0.1, 0.2],
                                [0.1, 0.5, 0.4],
                                [0.02, 0.9, 0.08]])

    y_true = np.array([[0,1,1]])
    
    act = Sigmoid_Activation()
    loss = BinaryCrossEntropyLoss()
    act.output = sigmoid_outputs
    
    loss.backward(sigmoid_outputs, y_true)
    output = act.backward(loss.dinput)
    print(output)  
    
    
    
    exit()
    
    X = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ], dtype=float)
    y = np.array([0, 1, 2])
    model = fitter.fit(X, y)
    predictions = fitter.predict(X)
    print(predictions)
    np.testing.assert_almost_equal(predictions,
                                   [[1, 0, 0],
                                    [0, 1, 0],
                                    [0, 0, 1]], decimal=3)
