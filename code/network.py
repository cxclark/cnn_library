import numpy as np
import utils as utils

class Network:
    '''
    Network manages a set of layers. 
    '''
    def __init__(self, layers = []):
        self.layers = layers

    def train(self, data, labels, learning_rate):
        """
        Trains the network.
        """
        # Iterate through the layers in forward propagation.
        # Compute the loss.
        # Iterate through the layers in backward propagation.












