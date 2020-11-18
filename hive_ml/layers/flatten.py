import numpy as np

class Flatten:
    """
    Implements flattening of convolution volumes before passing to dense layers.
    """
    def __init__(self):
        self.cache = {}

    def forward(self, Z):
        """
        Flattens volumes.
        Arguments:
            Z -- input, numpy 4D array of shape (batch_size, height, width, depth).
        Returns:
            data -- numpy array of shape ((height x width x depth), batch_size).
        """
        # Extract the shape of the input to the layer and save to variable.
        shape = Z.shape
        
        # Save the shape to the cache.
        self.cache['shape'] = shape

        # Reshape the input.
        data = Z.reshape(-1, shape[0])
        
        return data
 
    def backward(self, Z, lr):
        """
        Implements backpropagation of flatten layers. Reshapes volumes to forward propagation shapes.
        Arguments:
            Z -- numpy array of shape ((height x width x depth), batch_size).
        Returns:
            out -- numpy 4D array of shape (batch_size, height, width, depth).
        """
        # Transpose from ((height x width x depth), batch_size) to (batch_size, (height x width x depth)).
        Z = Z.T
        
        # Extract origincal shape from forward propagation step.
        shape = self.cache['shape']
        
        # Reshape volume to numpy 4D array of shape (batch_size, height, width, depth).
        out = Z.reshape(shape)

        return out