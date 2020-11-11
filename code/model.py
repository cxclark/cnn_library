import numpy as np

# Define a Model class to put together layers
class Model:
    '''
    Model manages a set of layers. 
    '''
    def __init__(self, *model, **kwargs):
        self.model = model
        self.num_classes = 0
        self.batch_size = 0
        self.loss = None
        self.optimizer = None
    
    # Define a function to set the batch size.
    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    # Define a function to set the loss.
    def set_loss(self, loss):
        self.loss = loss

    # Define a function to to train the network.
    def train(self, data, labels, batch_size=128, epochs=50,
            optimizer='adam'):

        self.set_batch_size(batch_size)

        iter = 1
        for epoch in range(epochs):
            print('Running epoch:', epoch + 1)
            for i, (x_batch, y_batch) in enumerate(get_batches(data, labels)):
                batch_preds = layer.forward_prop(batch_preds)
            dA = self.loss.compute_derivative(y_batch, batch_preds)
            for layer in reversed(self.model):
                dA = layer.backward_prop(dA)
                if layer.has_weights():
                    if optimization == 'adam':
                        layer.momentum()
                        layer.rmsprop()

                for layer in self.model:
                    if layer.has_weights():
                        layer.apply_grads(optimization=optimization, iter=iter)
            iter += batch_size

    def predict(self, data):
        if self.batch_size == 0:
            self.batch_size = data.shape[0]
        elif self.num_classes == 0:
            predictions = np.zeros((1, data.shape[0]))
        else:
            predictions = np.zeros((self.num_classes, data.shape[0]))
        num_batches = data.shape[0] // self.batch_size
        for batch_num, x_batch in enumerate(get_batches(data, batch_size=self.batch_size)):
            batch_preds = x_batch.copy()
            for layer in self.model:
                batch_preds = layer.forward_prop(batch_preds)
            M, N = predictions.shape[0]:
                predictions = np.zeros(shape=(M, data.shape[0]))
            if batch_num <= num_batches - 1:
                predictions[:, batch_num * self.batch_size:(batch_num + 1) * self.betch_size] = batch_preds
            else:
                predictions[:, batch_num * self.batch_size:] = batch_preds 

        return predictions

    def evaluate(self, data, labels):
        predictions = self.predict(data)
        return predictions 













