import numpy as np

class Flatten:
    def __init__(self):
        self.cache = {}

    def forward(self, Z):
        shape = Z.shape
        self.cache['shape'] = shape
        data = Z.reshape(-1, shape[0])
        
        return data
 
    def backward(self, Z, lr):
                
        Z = Z.T
        shape = self.cache['shape']
        out = Z.reshape(shape)
        return out