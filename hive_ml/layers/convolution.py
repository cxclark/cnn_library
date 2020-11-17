import numpy as np
import hive_ml.utils as utils

class ConvolutionLayer:
    def __init__(self, filters=3, filter_size=3, padding=1, stride=1):
        self.params = {
                'filters': filters,
                'filter_size': filter_size,
                'padding': padding,
                'stride': stride
                }
        self.cache = {}
        self.type = 'conv'

    def zero_pad(self, X, pad):
        """
        Pad all images in dataset X with zeros along height and width.
        Arguments:
            X -- python numpy array of shape (m, n_H, n_W, n_C) representing a batch of m images.
            pad -- zero-padding size to append to height and width.
        Returns:
            X_pad -- padded image of shape (m, n_H + 2*pad, n_W + 2*pad, n_C)
        """
        # Extract the padding value from the parameter dictionary.
        pad = self.params['padding']
        
        # Pad the images in dataset X.
        X_pad = np.pad(X, ((0,0), (pad, pad), (pad, pad), (0, 0)), mode='constant', constant_values=(0,0))
        return X_pad

    def conv_single_step(self, a_slice_prev, W, b):
        """
        Apply one filter on a single slice of the output activation of the previous layer.
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

    def forward(self, A_prev):
        """
        Implements forward propagation for a convolution layer.
        W -- weights, numpy array of shape (f, f, n_C_prev, n_C)
        b -- biases, numpy array of shape (1, 1, 1, n_C)
        Arguments:
            A_prev -- output activations of previous layer, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
        Returns:
            Z -- convolution output, numpy array of shape (m, n_H, n_W, n_C)
        """
        # Extract dimensions from A_prev's shape.
        m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape

        # Extract dimensions from the params dictionary.
        filter_size = self.params['filter_size']
        filters = self.params['filters']
        
        # Initialize a parameter matrix if it does not exist. 
        if 'W' not in self.params:
            W_shape = (filter_size, filter_size, n_C_prev, filters)
            b_shape = (1, 1, 1, filters)
            self.params['W'] = utils.layer_uniform(W_shape)
            self.params['b'] = utils.layer_uniform(b_shape)

        # Extract information from params dictionary.
        stride = self.params['stride']
        padding = self.params['padding']
        W = self.params['W']
        b = self.params['b']

        # Extract dimensions from W's shape.
        f, f, n_C_prev, n_C = W.shape

        # Compute the dimensions of the output volume.
        n_H = int((n_H_prev - f + 2*padding) / stride) + 1
        n_W = int((n_W_prev - f + 2*padding) / stride) + 1 

        # Initialize the output volume Z with zeros.
        Z = np.zeros([m, n_H, n_W, n_C])

        # Pad the input volume to the convolution.
        A_prev_pad = self.zero_pad(A_prev, padding)

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
                        horiz_end = horiz_start + f

                        # Use the corners to define the 3D slice of a_prev_pad.
                        # These should be the entire depth of the input layer.
                        a_slice_prev = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]

                        # Convolve the 3D slice with the correct filter W and bias b, return one output neuron.
                        Z[i, h, w, c] = self.conv_single_step(a_slice_prev, W[:, :, :, c], b[:, :, :, c])

        # Check to make sure your output shape is correct. 
        assert(Z.shape == (m, n_H, n_W, n_C))

        # Store information in the cache for backpropagation.
        self.cache['A_prev'] = A_prev
        self.params['W'] = W
        self.params['b'] = b

        return Z

    def backward(self, dZ, lr):
        """
        Implement the backward propagation for a convolution layer.
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
        # A_prev was the previous activation layer when doing forward propagation.
        # W and b were the weights and biases used to calculate the output when doing forward propagation.
        A_prev = self.cache['A_prev']
        W = self.params['W']
        b = self.params['b']

        # Extract dimensions from A_prev's shape.
        m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape 

        # Extract dimensions from W's shape.
        f, f, n_C_prev, n_C = W.shape

        # Extract hyperparameters.
        stride = self.params['stride']
        padding = self.params['padding']

        # Extract dimensions from dZ's shape.
        m, n_H, n_W, n_C = dZ.shape

        # Initialize dA_prev, dW, dB with the correct shapes.
        dA_prev = np.zeros((m, n_H_prev, n_W_prev, n_C_prev))
        dW = np.zeros((f, f, n_C_prev, n_C))
        db = np.zeros((1, 1, 1, n_C))

        # Pad A_prev and dA_prev.
        # These are used as the inputs and outputs of backpropagation.
        A_prev_pad = self.zero_pad(A_prev, padding)
        dA_prev_pad = self.zero_pad(dA_prev, padding)

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
                        a_prev_slice = a_prev_pad[vert_start:vert_end, horiz_start: horiz_end, :]

                        # Flow the gradient backward, update gradients for the defined slice.
                        da_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] += W[:,:,:,c] * dZ[i,h,w,c]
                        dW[:,:,:,c] += a_prev_slice * dZ[i,h,w,c]
                        db[:,:,:,c] += dZ[i,h,w,c]

            # Set the ith training example's dA_prev to the updated da_prev_pad.
            dA_prev[i,:,:,:] = da_prev_pad[padding:-padding, padding:-padding, :]

        # Check that your output shape is correct.
        assert(dA_prev.shape == (m, n_H_prev, n_W_prev, n_C_prev))

        # Update the parameters.
        self.params['W'] = W - lr * dW
        self.params['b'] = b - lr * db
        
        return dA_prev