import numpy as np

# Define a class to Flatten.
class Flatten:
    def __init__(self):
        self.shape = ()
        self.transpose = True
        self.cache = {}

    def forward(self, Z):
        shape = Z.shape
        self.cache['shape'] = shape
        data = np.ravel(Z).reshape(shape[0], -1)
        return data.T

    def backward(self, Z):
        Z = Z.T
        return Z.reshape(self.cache['shape'])
