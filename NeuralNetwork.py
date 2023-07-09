import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoid_derivative(x):
    return x * (1 - x)
def relu(x):
    return np.maximum(0, x)
def relu_derivative(x):
    return np.where(x > 0, 1, 0)
def softmax(x):
    x = x - np.max(x)  # for numerical stability
    return np.exp(x) / np.sum(np.exp(x), axis=0)
def softmax_derivative(x):
    sm = softmax(x)
    return sm * (1 - sm)
def cross_entropy(y_pred, y_true):
    n_samples = y_true.shape[0]
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    cross_entropy = -np.sum(y_true*np.log(y_pred))
    return cross_entropy / n_samples

# 3 Layers net
class NeuralNetwork:
    def __init__(self, input_dim=784, hidden_dim=800, output_dim=10, learning_rate=0.1, epochs=100,activation_function=sigmoid,activation_function_deriavte=sigmoid_derivative,lamda=0.001):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.activation_function=activation_function
        self.activation_function_deriavte=activation_function_deriavte
        self.l2_lambda=lamda

        # Xvier initialization
        self.W1 = np.random.randn(hidden_dim, input_dim) * np.sqrt(1. / hidden_dim)
        self.b1 = np.zeros((hidden_dim, 1))
        self.W2 = np.random.randn(output_dim, hidden_dim) * np.sqrt(1. / output_dim)
        self.b2 = np.zeros((output_dim, 1))

    def forward(self, X):
        self.Z1 = np.dot(X, self.W1.T) + self.b1.T
        self.A1 = self.activation_function(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2.T) + self.b2.T
        self.A2 = softmax(self.Z2)
        return self.A2

    def backward(self, X, Y):
        m = X.shape[0]
        # dZ2 is the derivative of the loss function (cross-entropy) with respect to Z2 (output before applying softmax)
        dZ2 = self.A2 - Y
        # dW2 is the derivative of the loss function with respect to W2 (weights of the second layer)
        # We're using the chain rule here: d_loss/d_W2 = d_loss/d_Z2 * d_Z2/d_W2,
        # where d_Z2/d_W2 = A1 due to the linearity of matrix multiplication in Z2 = W2*A1 + b2
        dW2 = (1 / m) * np.dot(dZ2.T, self.A1) + (self.l2_lambda / m) * self.W2
        # db2 is the derivative of the loss function with respect to b2 (biases of the second layer)
        # Again using the chain rule: d_loss/d_b2 = d_loss/d_Z2 * d_Z2/d_b2,
        # where d_Z2/d_b2 = 1
        db2 = (1 / m) * np.sum(dZ2, axis=0, keepdims=True)
        # dA1 is the derivative of the loss function with respect to A1 (output of the first layer)
        # Chain rule: d_loss/d_A1 = d_loss/d_Z2 * d_Z2/d_A1, where d_Z2/d_A1 = W2
        dA1 = np.dot(dZ2, self.W2)
        # dZ1 is the derivative of the loss function with respect to Z1 (output before applying activation_functionin the first layer)
        # Chain rule: d_loss/d_Z1 = d_loss/d_A1 * d_A1/d_Z1, where d_A1/d_Z1 is the derivative of activation_function
        dZ1 = dA1 * self.activation_function_deriavte(self.A1)
        # dW1 is the derivative of the loss function with respect to W1 (weights of the first layer)
        # Chain rule: d_loss/d_W1 = d_loss/d_Z1 * d_Z1/d_W1, where d_Z1/d_W1 = X
        dW1 = (1 / m) * np.dot(dZ1.T, X) +(self.l2_lambda / m) * self.W1
        # db1 is the derivative of the loss function with respect to b1 (biases of the first layer)
        # Chain rule: d_loss/d_b1 = d_loss/d_Z1 * d_Z1/d_b1, where d_Z1/d_b1 = 1
        db1 = (1 / m) * np.sum(dZ1, axis=0, keepdims=True)

        # Update weights and biases in the negative direction of the gradient
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1.T
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2.T

    def fit(self, X, Y):
        for i in range(self.epochs):
            self.forward(X)
            self.backward(X, Y)

    def predict(self, X):
        output = self.forward(X)
        return np.argmax(output, axis=1)

    def score(self, X, Y):
        predictions = self.predict(X)
        labels = np.argmax(Y, axis=1)
        return np.mean(predictions == labels)

# 4 Layers net
class NeuralNetwork_2:
    def __init__(self, input_dim=784, hidden_dim1=400, hidden_dim2=400, output_dim=10, learning_rate=0.01, epochs=3,
                 activation_function=sigmoid, activation_function_derivative=sigmoid_derivative,lamda=0):
        self.input_dim = input_dim
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.activation_function = activation_function
        self.activation_function_derivative = activation_function_derivative
        self.l2_lambda = lamda

        # Xvier initialization
        self.W1 = np.random.randn(hidden_dim1, input_dim) * np.sqrt(1. / hidden_dim1)
        self.b1 = np.zeros((hidden_dim1, 1))
        self.W2 = np.random.randn(hidden_dim2, hidden_dim1) * np.sqrt(1. / hidden_dim2)
        self.b2 = np.zeros((hidden_dim2, 1))
        self.W3 = np.random.randn(output_dim, hidden_dim2) * np.sqrt(1. / output_dim)
        self.b3 = np.zeros((output_dim, 1))
    def forward(self, X):
        self.Z1 = np.dot(X, self.W1.T) + self.b1.T
        self.A1 = self.activation_function(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2.T) + self.b2.T
        self.A2 = self.activation_function(self.Z2)
        self.Z3 = np.dot(self.A2, self.W3.T) + self.b3.T
        self.A3=softmax(self.Z3)
        return self.A3

    def backward(self, X, Y):
        m = X.shape[0]
        dZ3 = self.A3 - Y
        dW3 = (1 / m) * np.dot(dZ3.T, self.A2) +(self.l2_lambda / m) * self.W3
        db3 = (1 / m) * np.sum(dZ3, axis=0, keepdims=True)
        dA2 = np.dot(dZ3, self.W3)
        dZ2 = dA2 * self.activation_function_derivative(self.A2)
        dW2 = (1 / m) * np.dot(dZ2.T, self.A1)+(self.l2_lambda / m) * self.W2
        db2 = (1 / m) * np.sum(dZ2, axis=0, keepdims=True)
        dA1 = np.dot(dZ2, self.W2)
        dZ1 = dA1 * self.activation_function_derivative(self.A1)
        dW1 = (1 / m) * np.dot(dZ1.T, X) +(self.l2_lambda / m) * self.W1
        db1 = (1 / m) * np.sum(dZ1, axis=0, keepdims=True)

        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1.T
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2.T
        self.W3 -= self.learning_rate * dW3
        self.b3 -= self.learning_rate * db3.T

    def fit(self, X, Y):
        for i in range(self.epochs):
            self.forward(X)
            self.backward(X, Y)

    def predict(self, X):
        output = self.forward(X)
        return np.argmax(output, axis=1)

    def score(self, X, Y):
        predictions = self.predict(X)
        labels = np.argmax(Y, axis=1)
        return np.mean(predictions == labels)


















