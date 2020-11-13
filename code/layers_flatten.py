import numpy as np

class Flatten:
    def __init__(self):
        self.cache = {}

    def forward(self, Z):
        shape = Z.shape
        self.cache['shape'] = shape
        data = Z.flatten()
        return data.T

    def backward(self, Z, lr):
        Z = Z.T
        shape = self.cache['shape']
        return Z.reshape(shape)