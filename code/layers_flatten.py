import numpy as np

class Flatten:
    def __init__(self):
        self.cache = {}

    def forward(self, Z):
        shape = Z.shape
        self.cache['shape'] = shape
        data = np.ravel(Z).reshape(shape[0], -1)
        return data.T

    def backward(self, Z):
        Z = Z.T
        shape = self.cache['shape']
        return Z.reshape(shape)

    # Flatten layers have no parameters. 
    def update_params(self):
        pass 
