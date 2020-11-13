import numpy as np
import utils as utils

class Network:
    '''
    Network manages a set of layers. 
    '''
    def __init__(self, layers = []):
        self.layers = layers

    def predict(self, X):
        """
        Forward propagates through the network.
        """
        A = np.zeros(X.shape)
        for i in range(len(self.layers)):
            A = self.layers[i].forward(A)
        return A

    def evaluate(self, X, Y):
        """
        Evaluates predictions.
        """
        predictions = self.predict(X)
        # Extract the number of examples.
        m = A.shape[0]
        # Calculate the categorical cross-entropy
        cce = -np.sum(Y * np.log(predictions)) / m
        return (A, cce)

    def train(self, X, Y, learning_rate):
        """
        Trains the network.
        """
        for mini_batch in mini_batches:
            A, cce = self.evaluate(X, Y)
            dA = A - Y
            for i in reversed(range(len(self.layers))):
                dA = self.layers[i].backward(dA, learning_rate)
            return dA
        # Iterate through the layers in forward propagation.
        # Compute the loss.
        # Iterate through the layers in backward propagation.












