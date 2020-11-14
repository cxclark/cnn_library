import numpy as np

class Flatten:
    def __init__(self):
        self.cache = {}

    def forward(self, Z):
        shape = Z.shape
        self.cache['shape'] = shape
        data = Z.reshape(shape[0], -1)
        
        ### DEBUGGING ########################################################
        print(f'Z input shape in Flatten forward prop: {shape}')
        print(f'data output shape in Flatten forward prop: {data.shape}')
        
        return data.T
 
    def backward(self, Z, lr):
        Z = Z.T
        shape = self.cache['shape']
        return Z.reshape(shape)