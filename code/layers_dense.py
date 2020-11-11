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
        self.grads = {}
        self.type = 'fc'

    def forward(self, X):
        if 'W' not in self.params:
            self.params['W'], self.params['b'] = layer_init_uniform((X.shape[0], self-units))
        Z = np.dot(self.params['W'], X) + self.params['b']
        return Z

    def backward(self, dZ):
        batch_size = dZ.shape[1]
        self.grads['dW'] = np.dot(dZ, self.cache['A'].T) / batch_size
        self.grads['db'] = np.sum(dZ, axis=1, keepdims=True)
        return np.dot(self.params['W'].T, dZ)
