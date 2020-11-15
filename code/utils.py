import numpy as np
import pickle

cifar_classes = [
    'airplane', 
    'automobile', 
    'bird', 
    'cat', 
    'deer', 
    'dog', 
    'frog', 
    'horse', 
    'ship', 
    'truck'
]

def to_categorical(labels, num_classes):
    """
    Converts image labels to one-hot encoded binary vectors with length of num_classes.
    Arguments:
        labels -- image classification labels e.g. 1 = 'airplane', nested numpy arrays of integers.
        num_classes -- number of classes in image classification task, integer.
    Returns:
        label_vec -- binary vector representation of the class, 1 for correct label, 0's otherwise. 
    """
    # Initialize a zero matrix with the correct dimensions.
    label_vec = np.zeros((len(labels), num_classes))
    # Loop through the labels in the input.
    for i in range(len(labels)):
        # Create an new index for binary encoding, subtracting 1. 
        # An image with label [6] on a scale of 1-10, will have a 1 at index 5 in the vector.
        new_index = labels[i][0] - 1
        # Replace the 0 at the given location with a 1 for the correct class.
        label_vec[i][new_index] = 1
    return label_vec

def get_fs(shape):
    """
    Returns tuple of number of number of inputs to layer and number of outputs to layer.
    Arguments:
        shape -- numpy array.
    Returns:
        f_in, f_out -- tuple of integers.
    """
    # If the input shape comes from a fully-connected layer.
    if len(shape) == 2:
        f_in = shape[0]
        f_out = shape[1]
    # If the input shape comes from a convolutional layer with shape (m, n_H, n_W, c).
    else:
        # The number of inputs will be the product of n_H x n_W x c.
        f_in = np.prod(shape[1:])
        # The number of outputs will the number of examples m.
        f_out = shape[0]
    return f_in, f_out

def normal(shape, scale=0.05):
    """
    Random normal initialization.
    """
    return np.random.normal(0, scale, size=shape)

def uniform(shape, scale=0.05):
    """
    Random uniform initalization.
    """
    return np.random.uniform(-scale, scale, size=shape)


def layer_init_uniform(x):
    """
    Initialization inspired by tinygrad library.
    https://github.com/geohot/tinygrad/blob/master/tinygrad/utils.py
    """
    # Size is your output shape, int or tuple of ints
    ret = np.random.uniform(-1., 1., size=x)/np.sqrt(np.prod(x))
    return ret.astype(np.float32)

def he_normal(shape):
    """
    Initialization based on He et al. approach. 
    """
    fan_in, fan_out = get_fs(shape)
    scale = np.sqrt(2. / fan_in)
    shape = (fan_out, fan_in) if len(shape) == 2 else shape  # For a fully connected network
    # This supports only CNNs and fully connected networks
    bias_shape = (fan_out, 1) if len(shape) == 2 else (1, 1, 1, shape[3])   
    return normal(shape, scale), uniform(bias_shape)


def random_mini_batches(X, Y, mini_batch_size):
    """
    Creates a list of random minibatches from (X, Y)
    Arguments:
        X -- input data, of shape (m, n_H, n_W, c)
        Y -- true "label" vector of shape (m, num_classes)
        mini_batch_size -- size of mini-batches, integer

    Returns:
        mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    # Extract the input data shapes.
    m = X.shape[0]
    num_classes = Y.shape[1]
    
    # Instantiate an empty list to hold mini batch X-Y tuples with size batch_size.
    mini_batches = []

    # Shuffle X and Y.
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation, :, :, :]
    shuffled_Y = Y[permutation, :]
    
    ## DEBUGGING #####################################################################
    ### TOOK OUT .RESHAPE((1,m)) FROM DEEPLEARNING.AI, SHOULD PUT BACK IN?

    # Divide (shuffled_X, shuffled_Y) into batches minus the end case.
    num_complete_minibatches = m // mini_batch_size

    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[ k*mini_batch_size:(k+1)*mini_batch_size, :,:,:]
        mini_batch_Y = shuffled_Y[ k*mini_batch_size:(k+1)*mini_batch_size, :]

        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handle the end case if the last mini-batch < mini_batch_size.
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[ num_complete_minibatches*mini_batch_size: , :,:,:]
        mini_batch_Y = shuffled_Y[ num_complete_minibatches*mini_batch_size: , :]

        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches

def get_x_batches(X, mini_batch_size):
    """
    Creates a list of minibatches of X for predictions.
    Arguments:
        X -- input data, of shape (m, n_H, n_W, c)
        mini_batch_size -- size of mini-batches, integer

    Returns:
        mini_batches -- list of mini batches of input X.
    """
    # Extract the input data shapes.
    m = X.shape[0]
    
    # Instantiate an empty list to hold mini batch X-Y tuples with size batch_size.
    mini_batches_X = []

    # Divide X into batches minus the end case.
    num_complete_minibatches = m // mini_batch_size

    for k in range(0, num_complete_minibatches):
        mini_batch_X = X[ k*mini_batch_size:(k+1)*mini_batch_size, :,:,:]
        mini_batches_X.append(mini_batch_X)

    # Handle the end case if the last mini-batch < mini_batch_size.
    if m % mini_batch_size != 0:
        mini_batch_X = X[ num_complete_minibatches*mini_batch_size: , :,:,:]
        mini_batches_X.append(mini_batch_X)

    return mini_batches_X


# def export_model(model_name, filename):
#     pickle.dump(model_name, open(filename, 'wb'))
#     return
    
# def load_model(model_name, filename):
#     model_name = pickle.load(open(filename, 'rb'))
#     return



# def layer_init_uniform_dense(shape):
#     """
#     Initializes parameters using Xavier initialization.
#     """
#     # Size is your output shape, int or tuple of ints
#     f_in, f_out = get_fs(shape)
#     limit = np.sqrt(6 / float(f_in + f_out))
    
#     # If the input shape comes from a fully-connected layer.
#     if len(shape) == 2:
#         w_shape = (f_out, f_in)
#         b_shape = (f_out, 1)
        
#     # If the input shape comes from a convolutional or 3D input layer.
#     # W should have the shape (filter_size, filter_size, n_C_prev, filters).
#     # b should have the shape (1, 1, 1, filters).
#     else:
#         w_shape = shape
#         b_shape = (1, 1, 1, shape[3])
    
#     w = np.random.uniform(-limit, limit, size=w_shape)
#     b = np.random.uniform(-limit, limit, size=b_shape)
    
#     return w.astype(np.float32), b.astype(np.float32)

# def he_uniform_init(shape):
#     """
#     Initializes parameters. He et al. method common when using ReLU activations. 
#     """
#     f_in, f_out = get_fs(shape)
#     scale = np.sqrt(2. / f_in)
    
#     # If the input shape comes from a fully-connected layer.
#     if len(shape) == 2:
#         w_shape = (f_out, f_in)
#         b_shape = (f_out, 1)
        
#     # If the input shape comes from a convolutional or 3D input layer.
#     # W should have the shape (filter_size, filter_size, n_C_prev, filters).
#     # b should have the shape (1, 1, 1, filters).
#     else:
#         w_shape = shape
#         b_shape = (1, 1, 1, shape[3])
    
#     return uniform(w_shape, scale), uniform(b_shape)

# def xavier_init(shape):
#     """
#     Glorot/Xavier initialization as a function of input and output units. This version places
#     stricter limits on randomly generated numbers.
#     """
#     f_in, f_out = get_fs(shape)
#     limit = np.sqrt(6 / float(f_in + f_out))
#     return np.random.uniform(low=-limit, high=limit, size=(shape))


# def layer_init(shape):
#     """
#     Initializes weight numpy arrays with the correct shapes.
#     """
#     f_in, f_out = get_fs(shape)
#     limit = np.sqrt(6 / float(f_in + f_out))
    
#     # If the input shape comes from a fully-connected layer.
#     if len(shape) == 2:
#         w_shape = (f_out, f_in)
#         b_shape = (f_out, 1)
        
#     # If the input shape comes from a convolutional or 3D input layer.
#     # W should have the shape (filter_size, filter_size, n_C_prev, filters).
#     # b should have the shape (1, 1, 1, filters).
#     else:
#         w_shape = shape
#         b_shape = (1, 1, 1, shape[3])
    
#     w = np.random.uniform(low=-limit, high=limit, size=(w_shape))
#     b = np.random.uniform(low=-limit, high=limit, size=(b_shape))
    
#     return w, b 


# def xavier_initialization(shape):
#     f_in, f_out = get_fs(shape)
    
#     # Use Glorot/Xavier initialization
#     limit = np.sqrt(6 / float(f_in + f_out))    
   
#     # If the input shape comes from a fully-connected layer.
#     if len(shape) == 2:
#         w_shape = (f_out, f_in)
#         b_shape = (f_out, 1)
        
#     # If the input shape comes from a convolutional layer with shape (m, n_H, n_W, c).
#     else:
#         w_shape = shape
#         b_shape = (1, 1, 1, shape[3])
    
#     return np.random.uniform(low=-limit, high=limit, size=(f_in, f_out))
    
    
# # https://github.com/geohot/tinygrad/blob/master/tinygrad/utils.py
# def layer_init_uniform(x):
#     # Size is your output shape, int or tuple of ints
#     ret = np.random.uniform(-1., 1., size=x)/np.sqrt(np.prod(x))
#     return ret.astype(np.float32)


# def he_normal_init(shape):
#     '''
#     A function for smart normal distribution based initialization of parameters
#     [He et al. https://arxiv.org/abs/1502.01852]
#     :param fan_in: The number of units in previous layer.
#     :param fan_out: The number of units in current layer.
#     :return:[numpy array, numpy array]: A randomly initialized array of shape [fan_out, fan_in]
#     '''
#     fan_in, fan_out = get_fs(shape)
#     scale = np.sqrt(2. / fan_in)
#     shape = (fan_out, fan_in) if len(shape) == 2 else shape  # For a fully connected network
#     # This supports only CNNs and fully connected networks
#     bias_shape = (fan_out, 1) if len(shape) == 2 else (1, 1, 1, shape[3])   
#     return normal(shape, scale), uniform(bias_shape)

