import numpy as np
import hive_ml.utils as utils

class DenseLayer:
    """
    Computes dot products of dense or fully-connected layers.
    """
    def __init__(self, units=10):
        self.units = units
        self.params = {}
        self.cache = {}
        self.type = 'fc'

    def forward(self, X):
        """
        Implements forward propagation of dense layer.
        Arguments:
            X -- input data, of shape (64, 768)
        Returns:
            Z -- output scores to be passed to an activation function.
        """

        # Initialize a parameter matrix if it does not exist. 
        if 'W' not in self.params:
            self.params['W'], self.params['b'] = utils.he_normal((X.shape[0], self.units))
            
        W = self.params['W']
        b = self.params['b']

        # Save the input in the cache for backpropagation.
        self.cache['A'] = X
        
        Z = np.dot(W, X) + b

        return Z

    def backward(self, dZ, lr):
        batch_size = dZ.shape[1]
        self.cache['dW'] = np.dot(dZ, self.cache['A'].T) / batch_size
        self.cache['db'] = np.sum(dZ, axis=1, keepdims=True)
        
        # Extract the parameters.
        W = self.params['W']
        b = self.params['b']
        dW = self.cache['dW']
        db = self.cache['db']
        
        # Update parameters.
        self.params['W'] = W - lr * dW
        self.params['b'] = b - lr * db
        
        W = self.params['W']
        
        out = np.dot(W.T, dZ)
        
        return out