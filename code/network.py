import numpy as np
import utils as utils

class Network:
    '''
    Network manages a set of layers. 
    Parameters:
        num_classes -- number of classes, integer.
        batch_size -- size of mini-batches, integer.
        
    '''
    def __init__(self, layers = [], num_classes=10):
        self.layers = layers
        self.num_classes = num_classes

    def train(self, X, Y, learning_rate, batch_size=64, epochs=100):
        """
        Trains the network.
        Arguments:
            X -- dataset of images to train on, numpy array of shape (m, n_H, n_W, c).
            Y -- label vector containing true label images, numpy array of shape (m, 1).
            batch_size -- size of mini-batches, integer.
            learning_rate -- learning rate to use in updating parameters, scalar.
            epochs -- number of iterations through entire dataset to use in training, integer.
        Returns:
            None
        """        
        mini_batch_size = batch_size
        
        # Shuffle X and Y into random mini batches.
        mini_batches = utils.random_mini_batches(X, Y, mini_batch_size)
        
        # Loop through the epochs.
        for epoch in epochs:
            print('Running epoch:', epoch + 1)
            
            # Loop through the mini-batches.
            for mini_batch in mini_batches:
                    
                    # Extract mini_batch_X and save it as the input.
                    mini_batch_preds = mini_batch[0]
                    
                    #Loop through the layers in the network.
                    for layer in self.layers:
                        mini_batch_preds = layer.forward(mini_batch_preds)
                    
                    # Compute the loss.
                    dA = mini_batch[1] - mini_batch_preds
                    
                    # Loop through the reversed layers in the network.
                    for layer in reversed(self.layers):
                        dA = layer.backward(dA, learning_rate)

    def predict(self, X):
        """
        Forward propagates through the network.
        Arguments:
            X -- dataset of images, for which to predict labels, numpy array of shape (m, n_H, n_W, c).
        Returns:
            predictions -- label vector, numpy array of shape (m, 10).
        """
        # Initialize a numpy array for predictions of the correct shape.
        predictions = np.zeros(X.shape[0], self.num_classes)
        
        # Loop through the layers in the network.
        for layer in self.layers:
            predictions = layer.forward(predictions)
        
        return predictions

    
    def evaluate(self, X, Y):
        """
        Evaluates predictions.
        """
        predictions = self.predict(X)
        # Extract the number of examples.
        m = A.shape[0]
        # Calculate the categorical cross-entropy
        cce = -np.sum(Y * np.log(predictions)) / m
        return (A, cce)






