# Define a class to compute relu activations.
class ReluLayer:
    """
    Implements ReLU nonlinearity elementwise.
    x -> max(0, x)
    """

    def __init__(self):
        self.cache = {}
        self.layer_type = 'relu'

    def forward(self, Z):
        self.cache['Z'] = Z
        return np.where(Z < 0, 0, Z)

    def backward(self, dA):
        Z = self.cache['Z']
        return dA * np.where(Z < 0, 0, 1)

    def getParamsAndGrads(self):
        return []
