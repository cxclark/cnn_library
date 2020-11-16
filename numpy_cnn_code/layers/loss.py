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
        # Avoid the error by multiplying both the numerator and denominator with a constant.
        # A popular choice −max(z).
        
        expZ = np.exp(Z - np.max(Z))
        
        ### DEBUGGING ########################################################
        print(f'Z input shape in Softmax forward prop: {Z.shape}')
        
        out = expZ / expZ.sum(axis=0)

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

class CategoricalCrossEntropy:
    def compute_loss(labels, predictions):
        predictions = predictions / np.sum(predictions, axis=0, keepdims=True)
        return -np.sum(labels * np.log(predictions))

    def compute_derivative(labels, predictions):
        return labels - predictions 
