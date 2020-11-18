import numpy as np

class ReluLayer:
    """
    Implements ReLU nonlinearity elementwise.
    f(x) = max(0, x)
    """

    def __init__(self):
        self.cache = {}
        self.type = 'relu'

    def forward(self, Z):
        """
        Applies ReLU activation function to input Z.
        """
        # Save the input value for backpropagation.
        self.cache['Z'] = Z
        
        # Apply the relu activation to the input.
        return np.where(Z < 0, 0, Z)

    def backward(self, dA, lr):
        """
        Flows gradient dA back where values in forward propagation were non-negative.
        """
        # Extract the input value.
        Z = self.cache['Z']
        
        # Flow the gradient backward according to ReLU's derivative.
        return dA * np.where(Z < 0, 0, 1)