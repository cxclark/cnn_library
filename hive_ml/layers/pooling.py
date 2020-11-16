import numpy as np

class PoolLayer:
    """
    Pooling layer. May add average later.
    Max pooling finds areas of max activation.
    """
    def __init__(self, filter_size=2, stride=2, mode="max"):
        self.params = {
                'filter_size': filter_size,
                'stride': stride,
                'mode': mode
                }
        self.type = 'pooling'
        self.cache = {}

    def forward(self, A_prev):
        """
        Implements the forward pass of the pooling layer.
        Arguments:
            A_prev -- input data, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
        Returns:
            A -- output of the pooling layer, numpy array of shape (m, n_H, n_W, n_C)
        """
        # Extract dimensions from the input shape.
        m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape

        # Extract hyperparameters from the params dictionary.
        f = self.params['filter_size']
        stride = self.params['stride']

        # Define the dimensions of the output.
        n_H = int(1 + (n_H_prev - f) / stride)
        n_W = int(1 + (n_W_prev - f) / stride)
        n_C = n_C_prev

        # Initialize the output matrix A.
        A = np.zeros((m, n_H, n_W, n_C))

        # Loop over the training examples.
        for i in range(m):
            # Loop on the vertical axis of the output volume.
            for h in range(n_H):
                # Loop on the horizontal axis of the output volume.
                for w in range(n_W):
                    # Loop over the channels of the output volume.
                    for c in range(n_C):

                        # Find the corners of the current "slice".
                        vert_start = h * stride
                        vert_end = vert_start + f
                        horiz_start = w * stride
                        horiz_end = horiz_start + f

                        # Use the corners to define the current slice on the ith training example of A_prev, channel c.
                        a_prev_slice = A_prev[i, vert_start:vert_end, horiz_start:horiz_end, c]

                        # Compute the pooling operation on the slice
                        if self.params['mode'] == "max":
                            A[i, h, w, c] = np.max(a_prev_slice)

        # Store the input in the cache for backpropagation.
        self.cache['A_prev'] = A_prev

        # Check that your output shape is correct.
        assert(A.shape == (m, n_H, n_W, n_C))
        
        
        
        ### DEBUGGING ########################################################
        print(f'A_prev input shape in Pooling forward: {A_prev.shape}')
        print(f'A output shape in Pooling forkward: {A.shape}')

        
        
        return A

    def create_mask(self, X):
        """
        Creates a mask from an input matrix X, to identify the max entry of X. Used during backpropagation. 
        Arguments:
            X -- numpy array of shape (f, f)
        Returns:
            mask -- numpy array with same shape as window, contains True at position corresponding to max entry of X.
        """
        mask = X == np.max(X)
        return mask

    def backward(self, dA, lr):
        """
        Implements the backward pass of the pooling layer.
        Arguments:
            dA -- gradient of cost with respect to the output of the pooling layer, same shape as A.
        Returns:
            dA_prev -- gradient of cost with respect to the input of the pooling layer, same shape as A_prev.
        """
        # Extract information.
        A_prev = self.cache['A_prev']
        stride = self.params['stride']
        f = self.params['filter_size']

        # Extract dimensions from A_prev's shape and dA's shape.
        m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape
        m, n_H, n_W, n_C = dA.shape

        # Initialize dA_prev with zeros.
        dA_prev = np.zeros(A_prev.shape)

        # Loop over the training examples.
        for i in range(m):
            # Select training example from A_prev.
            a_prev = A_prev[i]
            # Loop on the vertical axis.
            for h in range(n_H):
                # Loop on the horizontal axis.
                for w in range(n_W):
                    # Loop over the channels (depth).
                    for c in range(n_C):

                        # Find the corners of the current "slice"
                        vert_start = h * stride
                        vert_end = vert_start + f
                        horiz_start = w * stride
                        horiz_end = horiz_start + f

                        # Compute the backward propagation.
                        if self.params['mode'] == "max":
                            # Use the corners and "c" to define the current slice from a_prev.
                            a_prev_slice = a_prev[vert_start:vert_end, horiz_start:horiz_end, c]

                            # Create the mask from a_prev_slice.
                            mask = self.create_mask(a_prev_slice)

                            # Set dA_prev to be dA_prev + (the mask multiplied by the correct entry of dA).
                            dA_prev[i, vert_start:vert_end, horiz_start:horiz_end, c] += np.multiply(mask, dA[i,h,w,c])

        # Check that your output shape is correct.
        assert(dA_prev.shape == A_prev.shape)
        
        
        
        ### DEBUGGING ########################################################
        print(f'A_prev cache input shape in Pooling backward: {A_prev.shape}')
        print(f'dA input shape in Pooling backward: {dA.shape}')
        print(f'dA_prev output shape in Pooling backward: {dA_prev.shape}')

        
        return dA_prev