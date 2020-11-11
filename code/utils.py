import numpy as np

def normal(shape, scale=0.05):
    return np.random.normal(0, scale, size=shape)

# https://github.com/geohot/tinygrad/blob/master/tinygrad/utils.py
def layer_init_uniform(x):
    # Size is your output shape, int or tuple of ints
    ret = np.random.uniform(-1., 1., size=x)/np.sqrt(np.prod(x))
    return ret.astype(np.float(32))
