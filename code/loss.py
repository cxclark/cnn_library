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
        out = np.exp(Z) / np.sum(np.exp(Z), axis=0)
        return out

    def backward(self, dA, lr):
        Z = self.cache['Z']
        return dA * (Z * (1 - Z))

    def update_params(self):
        pass

class CategoricalCrossEntropy:
    def compute_loss(labels, predictions):
        predictions = predictions / np.sum(predictions, axis=0, keepdims=True)
        return -np.sum(labels * np.log(predictions))

    def compute_derivative(labels, predictions):
        return labels - predictions 
