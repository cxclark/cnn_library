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
        labels -- image classification labels represented as index in a list, integer.
        num_classes -- number of classes in image classification task, integer.
    Returns:
        label_vec -- binary vector representation of the class, with 1 for correct label, 0's otherwise. 
    """
    # Initialize a zero matrix with the correct dimensions.
    label_vec = np.zeros(len(labels), int(num_classes))
    # Loop through the labels in the input.
    for label in labels:
        # Replace the 0 at the given location with a 1 for the correct class.
        # Subtract 1 to account for indexing
        label_vec[label - 1] = 1
    return label_vec

def normal(shape, scale=0.05):
    return np.random.normal(0, scale, size=shape)

def export_model(model_name, filename):
    pickle.dump(model_name, open(filename, 'wb'))
    return
    
def load_model(model_name, filename):
    model_name = pickle.load(open(filename, 'rb'))
    return

# https://github.com/geohot/tinygrad/blob/master/tinygrad/utils.py
def layer_init_uniform(x):
    # Size is your output shape, int or tuple of ints
    ret = np.random.uniform(-1., 1., size=x)/np.sqrt(np.prod(x))
    return ret.astype(np.float32)

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
    shuffled_Y = Y[permutation, :]
    
    ## TOOK OUT .RESHAPE((1,m)) FROM DEEPLEARNING.AI, SHOULD PUT BACK IN?

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
