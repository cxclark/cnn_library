import numpy as np

class SoftmaxLayer:
    """
    This is a classifier, with N discrete classes from 0 to N-1. The softmax function 
    receives scores and interprets them as the unnormalized log probabilities. It 
    exponentiates and normalizes the scores to produce probabilties.
    """
    def __init__(self):
        self.cache = {}

    def forward(self, Z):
        self.cache['Z'] = Z        
        
        # The Softmax function is not stable, will get floating point limitation error in NumPy.
        # Prone to overflow and underflow.
        # Overflow: when very large numbers are approximated as infinity
        # Underflow: when very small numbers are approximated as zero
        # To combat, common to shift input vector by subtracting the max element from all elements. 
        
        Z = Z - np.max(Z)
        expZ = np.exp(Z)
        
        ### DEBUGGING ########################################################
        print(f'expZ shape after np.exp(Z - np.max(Z)): {expZ.shape}')
        
        # Output has shape (num_classes, m).
        # Sum over axis 0 to normalize for each example. 
        # E.g. np.sum([[0, 1], [0, 5]], axis=0) -> array([0, 6])
        out = expZ / np.sum(expZ, axis=0, keepdims=True)

        ### DEBUGGING ########################################################
        print(f'Z input shape in Softmax forward prop: {Z.shape}')
        print(f'out output shape in Softmax forward prop: {out.shape}')

        return out
        
    def backward(self, dA, lr):
        Z = self.cache['Z']
        
        ### DEBUGGING ########################################################
        print(f'dA input shape in Softmax backward: {dA.shape}')
        print(f'Z input shape in Softmax backward: {Z.shape}')
        
        out = dA * (Z * (1 - Z))
        
        return out