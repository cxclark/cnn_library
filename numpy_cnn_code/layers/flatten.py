import numpy as np

class Flatten:
    def __init__(self):
        self.cache = {}

    def forward(self, Z):
        shape = Z.shape
        self.cache['shape'] = shape
        data = Z.reshape(-1, shape[0])
        
        ### DEBUGGING ########################################################
        print(f'Z input shape in Flatten forward prop: {shape}')
        print(f'data output shape in Flatten forward prop: {data.shape}')
        
        return data
        #return data.T
 
    def backward(self, Z, lr):
        
        ### DEBUGGING #####################################################
        print(f'Z input shape in Flatten backward: {Z.shape}')
        
        Z = Z.T
        
        ### DEBUGGING #####################################################
        print(f'Z transposed in Flatten backward: {Z.shape}')
        
        shape = self.cache['shape']
        
        out = Z.reshape(shape)
        
        ### DEBUGGING #####################################################
        print(f'out output shape in Flatten backward: {out.shape}')
        
        return out