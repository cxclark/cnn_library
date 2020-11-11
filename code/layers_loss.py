import numpy as np

"""
Layers that implement a loss.
"""

# Define a class to perform Softmax 
class SoftmaxLayer:
    """
    This is a classifier, with N discrete classes from 0 to N-1.
    
    The softmax function receives scores and interprets them as the 
    unnormalized log probailities. It exponentiates and normalizes the
    scores to produce probabilties.
    """
    def __init__(self):
        self.cache = {}

    def forward(self, Z,):
        self.cache['Z'] = Z
        out = np.exp(Z) / np.sum(np.exp(Z), axis=0)
        return out

    def backward(self, dA):
        Z = self.cache['Z']
        return dA * (Z * (1 - Z))

    def getParamsAndGrads(self):
        return []
