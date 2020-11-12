import numpy as np

# Define a class to complete convolutions.
class Convolution:
    def __init__(self, filters, filter_size, padding, stride):
        self.params = {
                'filters': filters,
                'filter_size': filter_size,
                'padding': padding,
                'stride': stride
                }
        self.cache = {}
        self.type = 'conv'
        self.grads = {}

    def conv_single_step(self, a_slice_prev, W, b):
        """
        Apply one filter on a single slice of the output activation
        of the previous layer.
        Arguments:
            a_slice_prev -- slice of input data of shape (f, f, n_C_prev)
            W -- weight parameters contained in a window, matrix of shape (f, f, n_C_prev)
            b -- bias parameters contained in a window, matrix of shape (1, 1, 1)
        Returns:
            Z -- a scaler value, result of convolving sliding window (W, b) on slice of input data.
        """
        c = np.multiply(a_slice_prev, W)
        Z = np.sum(c)
        Z = Z + np.float(b)

        return Z

    def zero_pad(self, X, pad):
        """
        Pad all images in dataset X with zeros along height and width.
        Arguments:
            X -- python numpy array of shape (m, n_H, n_W, n_C) representing a batch of m images.
        Returns:
            X_pad -- padded image of shape (m, n_H + 2*pad, n_W + 2*pad, n_C)
        """
        X_pad = np.pad(X, ((0,0), (pad, pad), (pad, pad), (0, 0)), mode='constant', constant_values=(0,0))
        return X_pad

    def forward(self, X, W, b, params):
        """
        Implements forward propagation for a convolution.
        Arguments:
            A_prev -- output activations of the previous layer, numpy array of 
                      shape (m, n_H_prev, n_W_prev, n_C_prev)
            W -- weights, numpy array of shape (f, f, n_C_prev, n_C)
            b -- biases, numpy array of shape (1, 1, 1, n_C)
        Returns:
            Z -- convolution output, numpy array of shape (m, n_H, n_W, n_C)
        """
        # Extract dimensions from A_prev's shape.
        m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape

        # Extract dimensions from W's shape.
        f, f, n_C_prev, n_C = W.shape

        # Extract information from params dictionary.
        stride = params['stride']
        padding = params['padding']
        
        # Compute the dimensions of the output volume.
        n_H = int((n_H_prev - f + 2*pad) / stride) + 1
        n_W = int((n_W_prev - f + 2*pad) / stride) + 1 

        # Initialize the output volume Z with zeros.
        Z = np.zeros([m, n_H, n_W, n_C])

        # Create A_prev_pad by padding A_prev.
        A_prev_pad = zero_pad(A_prev, padding)

        # Loop over the training examples.
        for i in range(m):
            # Select the ith training example's padded activation.
            a_prev_pad = A_prev_pad[i, :, :, :]
            # Loop over the vertical axis of the output volume.
            for h in range(n_H):
                # Loop over the horizontal axis of the output volume.
                for w in range(n_W):
                    # Loop over the channels of the output volume.
                    for c in range(n_C):

                        # Find the corners of the current "slice"
                        vert_start = h * stride
                        vert_end = vert_start + f
                        horiz_start = w * stride
                        horiz_end - horiz_start + f

                        # Use the corners to define the 3D slice of a_prev_pad.
                        a_slice_prev = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]

                        # Convolve the 3D slice with the correct filter W and bias b, return one output neuron.
                        Z[i, h, w, c] = conv_single_step(a_slice_prev, W[:, :, :, c], b[:, :, :, c])

        # Check to make sure your output shape is correct 
        assert(Z.shape == (m, n_H, n_W, n_C))

        self.cache['A_prev'] = A_prev
        self.cache['W'] = W
        self.cache['b'] = b

        return Z

    # Define a function for the backward convolution pass.
    def backward(self, dZ):
        """
        Implement the backward propagation for a convolution function.
        Arguments:
            dZ -- gradient of the cost with respect to the outout of conv layer (Z), 
                  numpy array of shape (m, n_H, n_W, n_C)
        Returns:
            dA_prev -- gradient of the cost with respect to the input of the conv layer (A_prev),
                       numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
            dW -- gradient of the cost with respect to the weights of conv layer (W),
                  numpy array of shape (f, f, n_C_prev, n_C)
            db -- gradient of the cost with respect to the biases of the conv layer (b), 
                  numpy array of shape (1, 1, 1, n_C)
        """
        # Extract information from the cache.
        A_prev = self.cache['A_prev']
        W = self.cache['W']
        b = self.cache['b']

        # Extract dimensions from A_prev's shape.
        m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape 

        # Extract dimensions fro W's shape.
        f, f, n_C_prev, n_C = W.shape

        # Extract hyperparameters.
        stride = params['stride']
        padding = params['padding']

        # Extract dimensions from dZ's shape.
        m, n_H, n_W, n_C = dZ.shape

        # Initialize dA_prev, dW, dB with the correct shapes.
        dA_prev = np.zeros((m, n_H_prev, n_W_prev, n_C_prev))
        dW = np.zeros((f, f, n_C_prev, n_C))
        db = np.zeros((1, 1, 1, n_C))

        # Pad A_prev and dA_prev.
        A_prev_pad = zero_pad(A_prev, padding)
        dZ_prev_pad = zero_pad(dA_prev, padding)

        # Loop over the training examples.
        for i in range(m):
            # Select the ith training example from A_prev_pad and dA_prev_pad.
            a_prev_pad = A_prev_pad[i]
            da_prev_pad = dA_prev_pad[i]

            # Loop over vertical axis of the output volume.
            for h in range(n_H):
                # Loop over horizontal axis of the output volume.
                for w in range (n_W):
                    # Loop over the channels of the output volume.
                    for c in range(n_C):

                        # Find the corners of the current "slice"
                        vert_start = h * stride
                        vert_end = vert_start + f
                        horiz_start = w * stride
                        horiz_end = horiz_start + f

                        # Use the corners to define the slice from a_prev_pad.
                        a_slice = a_prev_pad[vert_start:vert_end, horiz_start: horiz_end, :] += W[:,:,:,c] * dZ[i,h,w,c]
                        self.grads['dW'][:,:,:,c] += a_slice * dZ[i,h,w,c]
                        self.grads['db'][:,:,:,c] += dZ[i,h,w,c]

            # Set the ith training example's dA_prev to the updated da_prev_pad.
            dA_prev[i,:,:,:] = da_prev_pad[pad:-pad, pad:-pad, :]

        # Check that your output shape is correct.
        assert(dA_prev.shape == (m, n_H_prev, n_W_prev, n_C_prev))

        return dA_prev

    def update_params(self, lr):
        self.W = self.W - lr * self.dW
        self.b = self.b - lr * self.db

