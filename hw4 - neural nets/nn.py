import numpy as np
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold

np.random.seed(42)

class Layer:
    def __init__(self, input_dim, output_dim, lambda_, relu=False):
        self.input_dim = input_dim
        self.output_dim = output_dim  
        coef = 1 if not relu else 2
        self.weights = np.random.randn(input_dim, output_dim) * np.sqrt(coef / (input_dim)) # Xavier
        self.biases = np.ones((1, output_dim))
        self.lambda_=  lambda_
        self.reg_bias = np.random.rand() > 0.7  # If we want to regualrize biases
        
    def forward(self, X):
        self.X = X
        self.output = np.dot(X, self.weights) + self.biases     # XW + b
        return self.output
    
    def backward(self, d_inputs):
        self.dweights = np.dot(self.X.T, d_inputs)
        self.dbiases = np.sum(d_inputs, axis=0, keepdims=True)
        
        self.dweights += 2 * self.lambda_ * self.weights
        
        # Don't regularize bias
        # if self.reg_bias:
            # self.dbiases += 2 * self.lambda_ * self.biases
        
        self.dX = np.dot(d_inputs, self.weights.T)
        
        return self.dX
    
class Softmax:
    """Softmax activation function, used in the last layer - outputs probabilities
        Grad combined with loss in Softmax_CrossEntropyLoss
    """
    def forward(self, inputs):
        self.inputs = inputs 
        
        exp_inputs = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.probs = exp_inputs / np.sum(exp_inputs, axis=1, keepdims=True)
        
        return self.probs
    
    
class Loss:
    """Here just so we can inherit regularization"""
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
    def __init__(self, lr=0.01, decay=.9999):
        self.lr = lr
        self.decay = decay
        self.iter = 0
        
    def update_lr(self):
        self.lr *=  self.decay
        
    def update(self, layer):

        layer.weights -= self.lr * layer.dweights
        layer.biases -= self.lr * layer.dbiases
        return layer
    
    def update_iter(self):
        self.iter += 1


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
    def __init__(self, units, lambda_=0, n_iter=14000, verbose=False, activations=[], decay=1, lr=.1):
        np.random.seed(42)
        self.losses = []
        self.accuracies = []
        self.units = units
        self.lambda_ = lambda_
        self.softmax = Softmax()
        self.activations = activations
        self.loss = CategoricalCrossEntropyLoss(self.lambda_)
        self.softmax_cce = Softmax_CrossEntropyLoss(self.lambda_)
        self.optimizer = SGD_Optimizer(lr=lr, decay=decay)
        self.layers = []
        self.n_iter = n_iter
        self.verbose = verbose
        self.gradient_check = False
        self.epsilon = 1e-5
        
        
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
            self.optimizer.update_lr()              
            logits = self.forward_pass(X)
            
            rloss = self.regularization_loss()
            
            probs, loss = self.softmax_cce.forward(logits, y)
            loss += rloss
            
            preds = np.argmax(probs, axis=1)
            self.losses.append(loss)
            self.accuracies.append(np.mean(preds == y))
            acc = np.mean(preds == y)
            if i % 100 == 0: 
                if acc == 1 and loss < 0.001:
                    print("Converged after ", i, "iterations")
                    break
                
                if self.verbose:
                    print(f"Iteration: {i}")
                    print(f"Accuracy: {acc:.4f}")
                    print(f"Loss: {loss:.4f}")
                    print()
                    
            self.backward_pass(y)
            
            if self.gradient_check and i == 10:
                self.check_gradients(X, y, self.epsilon)
                
            self.update_parameters()   
            
            if i % 100 == 0:
                self.optimizer.update_iter()
            
        print(f"Accuracy: {acc:.4f}")
        print(f"Loss: {loss:.4f}")
            
        return self
        
    def build(self, X, y):
        self.units = [X.shape[1]] + self.units + [len(set(y))]
        print(self.units)

        for i in range(len(self.units) - 1):
            self.layers.append(Layer(self.units[i], self.units[i + 1], self.lambda_))

        if len(self.activations) == 0:
            self.activations = [Sigmoid_Activation() for _ in range(len(self.units) - 2)]
        
    def forward_pass(self, X):
        inp = X
        for i, layer in enumerate(self.layers[:-1]):
            Z = layer.forward(inp)
            inp = self.activations[i].forward(Z)

        logits = self.layers[-1].forward(inp)

        return logits
    
    def update_parameters(self):
        for layer in self.layers:
            self.optimizer.update(layer)
    
    def backward_pass(self, y):
        # Softmax + Loss 
        dL = self.softmax_cce.backward(y)
        
        # Last layer
        d = self.layers[-1].backward(dL)

        # Other layers
        for act, layer in zip((reversed(self.activations)), reversed(self.layers[:-1])):
            d = act.backward(d)
            d = layer.backward(d)

        
    
    def predict(self, X):
        logits = self.forward_pass(X)
        probs = self.softmax.forward(logits)
        return probs
        # predictions = np.argmax(probs, axis=1)
        # return np.eye(len(set(predictions)))[predictions] 

    def check_gradients(self, X, y, epsilon=1e-5):
        np.save_dir = "./" 

        for idx, layer in enumerate(self.layers):
            for param_name in ['weights', 'biases']:
                param = getattr(layer, param_name)
                grad = getattr(layer, 'd' + param_name)
                grad_copy = grad.copy()
                
                approx_grad = np.zeros_like(param)

                # Iterate over weights/biases
                it = np.nditer(param, flags=['multi_index'], op_flags=['readwrite'])    
                while not it.finished:
                    ix = it.multi_index
                    original_value = param[ix]

                    param[ix] = original_value + epsilon
                    _, plus_loss = self.softmax_cce.forward(self.forward_pass(X), y)

                    param[ix] = original_value - epsilon
                    _, minus_loss = self.softmax_cce.forward(self.forward_pass(X), y)

                    param[ix] = original_value
                    approx_grad[ix] = (plus_loss - minus_loss) / (2 * epsilon)

                    it.iternext()

                rel_error = np.abs(grad_copy - approx_grad) / (np.maximum(np.abs(grad_copy) + np.abs(approx_grad), 1e-8))
                max_error = np.max(rel_error)

                print(f"Layer {idx} {param_name} max relative error: {max_error:.2e}")

                np.save(f"{np.save_dir}layer{idx}_{param_name}_grad.npy", grad)
                np.save(f"{np.save_dir}layer{idx}_{param_name}_approx_grad.npy", approx_grad)


class ANNRegression(ANNClassification):
    def __init__(self, units, lambda_=0, lr=.1, n_iter=10000, verbose=False, activations=[]):
        np.random.seed(42)
        super().__init__(units, lambda_, n_iter, verbose, activations)
        self.lambda_ = lambda_
        self.loss = MeanSquaredErrorLoss(self.lambda_)
        self.optimizer = SGD_Optimizer(lr=lr)
        
    def weights(self):
        weights = []
        for layer in self.layers:
            weights.append(np.vstack([layer.weights, layer.biases]))
            
        return weights
        
    def fit(self, X, y):
        if y.ndim == 1:
            y = y.reshape(-1, 1)
            
        self.build(X, y)
        
        
        running_mean_loss = []
        
        for i in range(self.n_iter):
            self.optimizer.update_lr()
            
            logits = self.forward_pass(X)

            rloss = super().regularization_loss()

            loss = self.loss.forward(logits, y) + rloss
            running_mean_loss.append(loss)
            
            
            if i % 1000 == 0: 
                if np.mean(loss) < 0.0001:
                    print(f"Converged after {i} iterations")
                    break
                if self.verbose:
                    print(f"Loss: {np.mean(loss):.4f}")
                    print(self.optimizer.lr)
                    
            self.backward_pass(y, logits)
            super().update_parameters()
            
            if i % 100 == 0:
                self.optimizer.update_iter()
        return self
            
    def build(self, X, y):
        self.units = [X.shape[1]] + self.units + [1]
        self.layers = []
        for i in range(len(self.units) - 1):
            self.layers.append(Layer(self.units[i], self.units[i + 1], self.lambda_, relu=True))
            
        if len(self.activations) == 0:
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
        
        
    def predict(self, X):
        return self.forward_pass(X).flatten()
    
class ANN_Torch(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_units):
        super().__init__()
        
        # A single hidden layer
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_dim, hidden_units[0]),
            nn.Sigmoid(),
            nn.Linear(hidden_units[0], hidden_units[1]),
            nn.Sigmoid(),
            nn.Linear(hidden_units[1], output_dim)
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits
    
    
def train_torch(clf, optimizer, X, y, n_iter):
    
    X = torch.from_numpy(X)
    y = torch.from_numpy(y)
    
    criterion = nn.CrossEntropyLoss()
    for i in range(n_iter):
        clf.train()
        optimizer.zero_grad()
        
        logits = clf(X.float())
        loss = criterion(logits, y.long())
        
        loss.backward()
        if i == 10:
            for idx, layer in enumerate(clf.linear_relu_stack):
                if isinstance(layer, torch.nn.Linear):
                    for param_name, param in [('weights', layer.weight), ('biases', layer.bias)]:
                        grad = param.grad.detach().cpu().numpy()
                        np.save(f'torch_layer{idx}_{param_name}_grad.npy', grad)

        optimizer.step()
        
        print(f"Epoch {i+1}/{n_iter}, Loss: {loss.item():.4f}")
        
    return clf

        

if __name__ == "__main__":

    n_iter = 1000
    lr = 1e-3
    

    # Define model and data
    fitter = ANNClassification(units=[3, 3], n_iter=n_iter, lr=lr, verbose=False, lambda_=0)
    X, y = squares()
    X, y = doughnut()

    # Shuffle once at the beginning
    idxs = np.random.permutation(len(X))
    X = X[idxs]
    y = y[idxs]

    # Setup 10-fold CV
    kf = KFold(n_splits=10)

    ann_accs = []
    torch_accs = []
    ann_losses = []
    torch_losses = []

    for train_idx, test_idx in kf.split(X):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        
        # ANNClassification
        fitter = ANNClassification(units=[3, 3], n_iter=n_iter, lr=lr, verbose=False, lambda_=0)
        fitter.fit(X_train, y_train)
        probs = fitter.predict(X_test)
        
        # Torch Model
        torch_fitter = ANN_Torch(X.shape[1], len(np.unique(y)), [3,3])
        optimizer = optim.SGD(torch_fitter.parameters(), lr=lr)
        torch_clf = train_torch(torch_fitter, optimizer, X_train, y_train, n_iter)
        
        torch_clf.eval()
        logits = torch_clf(torch.from_numpy(X_test).float())
        softmax = nn.Softmax(dim=1)
        tprobs = softmax(logits)
        
        # Losses and accuracies
        celoss = CategoricalCrossEntropyLoss(0)
        
        ann_loss = celoss.forward(probs, y_test)
        torch_loss = celoss.forward(tprobs.detach().numpy(), y_test)
        ann_acc = np.mean(np.argmax(probs, axis=1) == y_test)
        torch_acc = np.mean(np.argmax(tprobs.detach().numpy(), axis=1) == y_test)
        
        ann_losses.append(ann_loss)
        torch_losses.append(torch_loss)
        ann_accs.append(ann_acc)
        torch_accs.append(torch_acc)
        break
    # Summary
    print(f"ANN mean accuracy: {np.mean(ann_accs):.4f}")
    print(f"Torch mean accuracy: {np.mean(torch_accs):.4f}")
    print(f"ANN mean loss: {np.mean(ann_losses):.4f}")
    print(f"Torch mean loss: {np.mean(torch_losses):.4f}")


    exit()
    # np.save("losses.npy", fitter.losses)
    # np.save("accuracies.npy", fitter.accuracies)
    pred = fitter.predict(X)
    np.save("exp.npy", pred)
    
    print(pred, y)
    # np.mean(np.argmax(pred, axis=1) == y)
    
    exit()
    
    # np.save("fitted_cos.npy", fitter.layers[-1].output)
    
    
    # m = fitter.fit(X_train, y_train)
    # exit()

    
    
    X = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ], dtype=float)
    y = np.array([0, 1, 2])
    
    # X,y = doughnut()
    # print(X.shape, y.shape)
    
    
    # exit()
    model = fitter.fit(X, y)
    predictions = fitter.predict(X)
    print(predictions)
    np.testing.assert_almost_equal(predictions,
                                   [[1, 0, 0],
                                    [0, 1, 0],
                                    [0, 0, 1]], decimal=3)
