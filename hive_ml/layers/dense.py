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
            X -- input data, numpy array of shape (flattened_neurons, batch_size).
            W -- weights, numpy array of shape (num_classes, flattened_neurons). Extracted from self.params.
            b -- biases, numpy array of shape (num_classes, 1). Extracted from self.params.
        Returns:
            Z -- output scores to be passed to an activation function, numpy array of shape (num_classes, batch_size).
        """

        # Initialize a parameter matrix if it does not exist. 
        if 'W' not in self.params:
            self.params['W'], self.params['b'] = utils.he_normal((X.shape[0], self.units))

        # Extract W and b values and save to variables.
        W = self.params['W']
        b = self.params['b']

        # Save the input in the cache for backpropagation.
        self.cache['A'] = X
        
        # Compute the dot product and add the bias.
        Z = np.dot(W, X) + b

        return Z

    def backward(self, dZ, lr):
        """
        Implements backward propagation of dense layer.
        """
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