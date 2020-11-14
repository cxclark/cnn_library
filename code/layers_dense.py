import numpy as np
import utils as utils

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
        """
        Implements forward propagation of dense layer.
        Arguments:
            X -- input data, of shape (64, 768)
        """

        # Initialize a parameter matrix if it does not exist. 
        if 'W' not in self.params:
            W_shape = X.shape[1]
            b_shape = self.units
            self.params['W'] = utils.layer_init_uniform(W_shape)
            self.params['b'] = utils.layer_init_uniform(b_shape)
        W = self.params['W']
        b = self.params['b']

        # Save the input in the cache for backpropagation.
        self.cache['A'] = X

        ### DEBUGGING
        print(f'W input shape in Dense forward prop: {W.shape}')
        print(f'X input shape in Dense forward prop: {X.shape}')
        print(f'b input shape in Dense forward prop: {b.shape}')
        
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

        return np.dot(W.T, dZ)