import numpy as np
import utils

class DenseLayer:
    """
    Fully connected dot products. 
    """
    def __init__(self, units=100):
        self.units = units
        self.params = {}
        self.cache = {}
        self.type = 'fc'

    def forward(self, X):

        # Initialize a parameter matrix if it does not exist. 
        if 'W' not in self.cache:
            W_shape = X.shape[0]
            b_shape = self.units
            self.cache['W'] = layer_init_uniform(W_shape)
            self.cache['b'] = layer_init_uniform(b_shape)
        W = self.cache['W']
        b = self.cache['b']

        # Save the input in the cache for backpropagation.
        self.cache['dense_input'] = X

        Z = np.dot(W, X) + b

        return Z

    def backward(self, dZ, lr):
        batch_size = dZ.shape[1]
        self.cache['dW'] = np.dot(dZ, self.cache['A'].T) / batch_size
        self.cache['db'] = np.sum(dZ, axis=1, keepdims=True)
        W = self.cache['W']
        return np.dot(W.T, dZ)

    def update_params(self, lr):
        W = self.cache['W']
        b = self.cache['b']
        dW = self.cache['dW']
        db = self.cache['db']

        self.cache['W'] = W - lr * dW
        self.cache['b'] = b - lr * db
