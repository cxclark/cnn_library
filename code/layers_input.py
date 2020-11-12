import numpy as np

# Define a class to control inputs into the network.
class InputLayer:
    def __init__(self):
        self.layer_type = 'input'

    def forward(self, X):
        return X

    def backward(self):
        pass

    def update_params(self):
        pass

