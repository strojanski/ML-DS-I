import numpy as np
import csv

class Layer:
    def __init__(self, input_dim, output_dim, lambda_):
        self.input_dim = input_dim # of features (from prev layer)
        self.output_dim = output_dim  # of neurons
        self.weights = np.random.randn(input_dim, output_dim) * np.sqrt(1. / (input_dim))*2 # Xavier
        self.biases = np.ones((1, output_dim))
        self.lambda_=  lambda_
        self.reg_bias = np.random.rand() > 0.7
        
    def forward(self, X):
        self.X = X
        self.output = np.dot(X, self.weights) + self.biases
        return self.output
    
    def backward(self, d_inputs):
        self.dweights = np.dot(self.X.T, d_inputs)
        self.dbiases = np.sum(d_inputs, axis=0, keepdims=True)
        
        self.dweights += 2 * self.lambda_ * self.weights
        
        if self.reg_bias:
            self.dbiases += 2 * self.lambda_ * self.biases
        
        self.dX = np.dot(d_inputs, self.weights.T)
        
        return self.dX
    
class Softmax:
    """Softmax activation function, used in the last layer - outputs probabilities
        Grad combined with loss in Softmax_CrossEntropyLoss
    """
    def forward(self, inputs):
        self.inputs = inputs 
        
        exp_inputs = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probs = exp_inputs / np.sum(exp_inputs, axis=1, keepdims=True)
        
        self.probs = probs
        return probs
    
    
class Loss:
    def __init__(self, lambda_):
        self.lambda_ = lambda_
        
    def regularize_bias(self, layer: Layer):
        if not layer.reg_bias:
            return 0
        return self.lambda_ * np.sum(layer.biases**2)
    
    def regularize_weights(self, layer: Layer):
        return self.lambda_ * np.sum(layer.weights**2)
    
        

class Sigmoid_Activation:
    """Sigmoid activation function, used in between hidden layers"""
    def __init__(self):
        self.inputs = None
        self.output = None
        self.dinput = None
        
    def forward(self, inputs):
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-1 * inputs))
        return self.output
    
    def backward(self, d_inputs):
        self.dinput = d_inputs * self.output * (1 - self.output)
        return self.dinput
    
class ReLU_Activation:
    
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
        return self.output
    
    def backward(self, d_inputs):
        self.dinput = d_inputs.copy()
        self.dinput[self.inputs <= 0] = 0
        return self.dinput
    
class LeakyReLU_Activation:
    def forward(self, x):
        self.input = x
        return np.where(x > 0, x, 0.01 * x)

    def backward(self, d_out):
        self.d_out = d_out * np.where(self.input > 0, 1, 0.01) 
        return self.d_out
    
class CategoricalCrossEntropyLoss(Loss):
    """Categorical Cross-Entropy loss function, as the loss function (global)"""
    def __init__(self, lambda_):
        super().__init__(lambda_)
        
    def forward(self, y_pred, y_true):
        """y_pred = softmax output, y_true = labels, need to one-hot encoded (done in this funciton)"""
        n_samples = len(y_true)
        
        y_pred = np.clip(y_pred, 1e-8, 1 - 1e-8)  # Avoid log(0)
        
        confidences = y_pred[range(n_samples), y_true]
        
        # Onehot encode
        if len(y_true.shape) == 2:
            confidences = np.sum(y_pred * y_true, axis=1)
            
        neg_log_likelihood = -np.log(confidences)
        return np.mean(neg_log_likelihood)
    

class Softmax_CrossEntropyLoss:
    def __init__(self, lambda_):
        self.activation = Softmax()
        self.loss = CategoricalCrossEntropyLoss(lambda_)
        
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
    
class Linear_Activation:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = inputs
        return self.output
        
    def backward(self, d_inputs):
        self.dinputs = d_inputs
        return self.dinputs
        
        
class MeanSquaredErrorLoss(Loss):
    def __init__(self, lambda_):
        super().__init__(lambda_)
    
    
    def forward(self, y_pred, y_true):
        return np.mean((y_pred - y_true) ** 2)
    
    def backward(self, y_pred, y_true):
        return 2 * (y_pred - y_true) / y_pred.shape[0]
    
class SGD_Optimizer:
    def __init__(self, lr=0.01):
        self.lr = lr
        
    def update(self, layer):
        np.clip(layer.dweights, -1, 1, out=layer.dweights)
        np.clip(layer.dbiases, -1, 1, out=layer.dbiases)

        layer.weights -= self.lr * layer.dweights
        layer.biases -= self.lr * layer.dbiases
        return layer


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
        self.lambda_ = lambda_
        self.softmax = Softmax()
        self.loss = CategoricalCrossEntropyLoss(self.lambda_)
        self.softmax_cce = Softmax_CrossEntropyLoss(self.lambda_)
        self.optimizer = SGD_Optimizer(lr=1)
        self.layers = []
        self.n_iter = n_iter
        self.verbose = verbose
        
    def regularization_loss(self):
        loss = 0
        
        for layer in self.layers:
            wloss = self.loss.regularize_weights(layer)
            bloss = self.loss.regularize_bias(layer)
            loss = loss + wloss + bloss
            
        return loss
        
        
    def fit(self, X, y):
        self.build(X, y)
        
        for i in range(self.n_iter):
            # Decay
            if i % 1000 == 0:
                self.optimizer.lr *= 0.999
            
            logits = self.forward_pass(X)
            
            rloss = self.regularization_loss()
            
            probs, loss = self.softmax_cce.forward(logits, y)
            loss += rloss
            
            preds = np.argmax(probs, axis=1)
            acc = np.mean(preds == y)
            if i % 100 == 0: 
                if acc == 1 and loss < 0.01:
                    print("Converged after ", i, "iterations")
                    break
                
                if self.verbose:
                    print(f"Accuracy: {acc:.4f}")
                    print(f"Loss: {loss:.4f}")
                    
            
            self.backward_pass(y)
            
        print(f"Accuracy: {acc:.4f}")
        print(f"Loss: {loss:.4f}")
            
        return self
        
    def build(self, X, y):
        self.units = [X.shape[1]] + self.units + [len(set(y))]
        print(self.units)

        for i in range(len(self.units) - 1):
            self.layers.append(Layer(self.units[i], self.units[i + 1], self.lambda_))

        self.activations = [Sigmoid_Activation() for _ in range(len(self.units) - 2)]

        
    def forward_pass(self, X):
        inp = X
        for i, layer in enumerate(self.layers[:-1]):
            Z = layer.forward(inp)
            inp = self.activations[i].forward(Z)

        logits = self.layers[-1].forward(inp)

        return logits
    
    def backward_pass(self, y):
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


class ANNRegression(ANNClassification):
    def __init__(self, units, lambda_=0, n_iter=10000, verbose=True):
        super().__init__(units, lambda_, n_iter, verbose)
        self.lambda_ = lambda_
        self.loss = MeanSquaredErrorLoss(self.lambda_)
        self.optimizer = SGD_Optimizer(lr=1)
        
    def fit(self, X, y):
        if y.ndim == 1:
            y = y.reshape(-1, 1)
            
        self.build(X, y)
        
        running_mean_loss = []
        
        for i in range(self.n_iter):
            # Update optimizer's learning rate with decay
            if i % 200 == 0:
                self.optimizer.lr = np.maximum(1 * 0.99**i, 1e-4)
                # print(self.optimizer.lr)
            
            logits = self.forward_pass(X)

            rloss = super().regularization_loss()

            loss = self.loss.forward(logits, y) + rloss
            running_mean_loss.append(loss)
            
            
            if i % 100 == 0: 
                if np.mean(loss) < 0.1:
                    print(f"Converged after {i} iterations")
                    break
                if self.verbose:
                    print(f"Loss: {np.mean(loss):.4f}")
                    
            self.backward_pass(y, logits)
            # for layer in self.layers:
            #     print("Weights std:", np.std(layer.weights))
            #     print("Biases std:", np.std(layer.biases))
            #     print("dWeights std:", np.std(layer.dweights))
            #     print("dBiases std:", np.std(layer.dbiases))

        return self
            
    def build(self, X, y):
        self.units = [X.shape[1]] + self.units + [1]
        self.layers = []
        for i in range(len(self.units) - 1):
            self.layers.append(Layer(self.units[i], self.units[i + 1], self.lambda_))
            
        self.activations = [LeakyReLU_Activation() for _ in range(len(self.units) - 2)]
            
    def forward_pass(self, X):
        inp = X
        for i, layer in enumerate(self.layers[:-1]):
            inp = layer.forward(inp)
            inp = self.activations[i].forward(inp)
        logits = self.layers[-1].forward(inp)
        
        return logits
            
            
    def backward_pass(self, y, logits):
        d_loss = self.loss.backward(logits, y)
        d = d_loss
        
        d = self.layers[-1].backward(d)

        for act, layer in zip(reversed(self.activations), reversed(self.layers[:-1])):
            d = act.backward(d)
            d = layer.backward(d)
        
        for layer in self.layers:
            self.optimizer.update(layer)     
        
    def predict(self, X):
        return self.forward_pass(X).flatten()
    

if __name__ == "__main__":

    fitter = ANNClassification(units=[5,5,5], lambda_=0., verbose=True)
    # fitter = ANNRegression(units=[10,10,10], lambda_=.1, verbose=True)
    X, y = squares()

    # Create X with values from 1 to 10 with a step of 0.01
    # X = np.arange(1, 10, 0.01).reshape(-1, 1)

    # Create y as the sine of each value in X
    # y = np.cos(X).reshape(-1, 1)
    
    fitter.fit(X, y)
    # X, y = doughnut()
    
    np.save("fitted_cos.npy", fitter.layers[-1].output)
    
    
    # m = fitter.fit(X_train, y_train)
    exit()
    # pred = m.predict(X_test)
    
    # pred = np.argmax(pred, axis=1)
    # print("Accuracy on test set:", np.mean(pred == y_test))
    
    # exit()
    
    
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
