import numpy as np

def normal(shape, scale=0.05):
    return np.random.normal(0, scale, size=shape)

# https://github.com/geohot/tinygrad/blob/master/tinygrad/utils.py
def layer_init_uniform(x):
    # Size is your output shape, int or tuple of ints
    ret = np.random.uniform(-1., 1., size=x)/np.sqrt(np.prod(x))
    return ret.astype(np.float(32))

def random_mini_batches(X, Y, mini_batch_size=64):
    """
    Creates a list of random minibatches from (X, Y)
    Arguments:
        X -- input data, of shape (m, n_H, n_W, c)
        Y -- true "label" vector of shape (m, 1)
        mini_batch_size -- size of mini-batches, integer

    Returns:
        mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """

    m = X.shape[0]
    mini_batches = []

    # Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation, :, :, :]
    shuffled_Y = Y[permutation, :].reshape((1, m))

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
        mini_batch_Y = shuffled_Y[ num_complete_minibatches*mini_batch_size: , :,:,:]

        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches
