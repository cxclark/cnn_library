import numpy as np
import hive_ml.utils as utils

class Model:
    '''
    Network manages a set of layers. 
    Parameters:
        num_classes -- number of classes, integer.
        batch_size -- size of mini-batches, integer.
        
    '''
    def __init__(self, *model, **kwargs):
        """
        * (single asterisk) and ** (double asterisks) lets you pass a variable number of arguments to function.
        *args is used to pass a non-keyworded variable-length argument list to your function. 
        **kwargs lets you pass a keyworded variable-length of arguments to your function
        """
        self.model = model
        self.num_classes = 10
        self.batch_size = 0
        
    def set_batch_size(self, batch_size):
        """
        Sets a batch size. This will be called while training the model.
        """
        self.batch_size = batch_size

    def train(self, X, Y, learning_rate, batch_size, epochs):
        """
        Trains the network.
        Arguments:
            X -- dataset of images to train on, numpy array of shape (m, n_H, n_W, c).
            Y -- label vector containing true label images, numpy array of shape (m, num_classes).
            batch_size -- size of mini-batches, integer.
            learning_rate -- learning rate to use in updating parameters, scalar.
            epochs -- number of iterations through entire dataset to use in training, integer.
        Returns:
            None
        """      
        # Call the set_batch_size function and pass it the batch_size argument.
        self.set_batch_size(batch_size)
        
        # Normalize the input data.
        X = X / 255
        
        # Convert labels to binary vector representations of the correct class.
        Y = utils.to_categorical(Y, self.num_classes)
        
        # Check that Y has the shape (m, num_classes).
        assert(Y.shape == (Y.shape[0], self.num_classes))
                
        # Shuffle X and Y into random mini batches.
        mini_batches = utils.random_mini_batches(X, Y, batch_size)
        
        # Loop through the epochs.
        for epoch in range(epochs):
            print('Running epoch:', epoch + 1)
            
            # Loop through the mini-batches.
            for mini_batch in mini_batches:
                    
                    # Extract mini_batch_X and save it as the input.
                    mini_batch_preds = mini_batch[0]
                    # Extract mini_batch_X true labels.
                    mini_batch_true_labels = mini_batch[1]
                    
                    #Loop through the layers in the network.
                    for layer in self.model:
                        mini_batch_preds = layer.forward(mini_batch_preds)

                    # Compute the derivative.
                    dA = mini_batch_preds - mini_batch_true_labels.T
                    
                    # Loop through the reversed layers in the network.
                    for layer in reversed(self.model):
                        dA = layer.backward(dA, learning_rate)
            
            # Compute the Categorical CrossEntropy loss after each epoch.
            probabilities = self.predict(X)
            probabilities = probabilities.T
            loss = -np.sum(Y * np.log(probabilities + 1e-8))
            m = X.shape[0]
            print(f'Loss epoch {epoch + 1}: {round(loss / m, 3)}')
                        
            # Compute the accuracy.
            # np.argmax() returns the indices of the maximum values along an axis.
            Y_hat_temp = np.argmax(probabilities, axis=1)
            Y_temp = np.argmax(Y, axis=1)            
            accuracy = (Y_hat_temp == Y_temp).mean()
            accuracy = round(accuracy * 100, 3)               
            print(f'Accuracy epoch {epoch + 1}: {accuracy}%')            

    def predict(self, X):
        """
        Forward propagates through the network and makes predictions on input dataset X.
        Arguments:
            X -- dataset of images, for which to predict labels, numpy array of shape (m, n_H, n_W, c).
        Returns:
            predictions -- label vector, numpy array of shape (m, 10).
        """
        # Normalize the input data.
        X = X / 255
        
        # Copy input X and save to variable x_preds.
        x_preds = X.copy()
        
        # Iterate through the model layers.
        for layer in self.model:
            x_preds = layer.forward(x_preds)

        return x_preds

    def evaluate(self, X, Y):
        """
        Evaluates predicted labels against true labels.
        Arguments:
            X -- dataset of images, numpy array of shape (m, n_H, n_W, c).
            Y -- true labels, numpy array of shape (m, 1).
        Returns:
            
        """
        # Normalize the image training data. 
        X = X / 255
        
        # Convert labels to binary vector representations of the correct class.
        # Output shape will be (num_classes, m).
        Y = utils.to_categorical(Y, self.num_classes)
        
        # Calculate the vector probabilities.
        # Output shape should be (num_classes, m).
        probabilities = self.predict(X)
        
        # Reshape probabilities from (num_classes, m) to (m, num_classes). 
        # The probability vector for every example should add up to 1.
        probabilities = probabilities.T                
        
        # Compute the accuracy.
        # np.argmax() returns the indices of the maximum values along an axis.
        Y_hat = np.argmax(probabilities, axis=1)
        Y = np.argmax(Y, axis=1)            
        accuracy = (Y_hat == Y).mean()
        accuracy = round(accuracy * 100, 3)               
        print(f'Accuracy: {accuracy}%')  
                
        return