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
        * (single asterisk) and ** (double asterisks) let you pass a variable number of arguments to a function.
        *args is used to pass a non-keyworded variable-length argument list to your function. 
        **kwargs lets you pass a keyworded variable-length of arguments to your function
        """
        self.model = model
        self.num_classes = 10
        self.batch_size = 0

    def train(self, X, Y, learning_rate, batch_size=64, epochs=100):
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
        # Assign batch size given during training to model batch_size.
        self.batch_size = batch_size
        
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
            print(f'Loss epoch {epoch + 1}: {round(loss, 3)}')
                        
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
        
        # Extract the input data shapes.
        m = X.shape[0]
        
        # Initialize a numpy array for predictions of the correct shape.
        predictions = np.zeros((self.num_classes, m))
        
        # Divide X into batches minus the end case.
        num_complete_minibatches = m // self.batch_size

        for k, mini_batch_X in enumerate(utils.get_x_batches(X, self.batch_size)):
            
            mini_batch_preds = mini_batch_X.copy()
            
            #Loop through the layers in the network.
            for layer in self.model:
                mini_batch_preds = layer.forward(mini_batch_preds)
            
            if k <= num_complete_minibatches - 1:
                predictions[:, k*self.batch_size:(k+1)*self.batch_size] = mini_batch_preds
            else:
                predictions[:, k*self.batch_size: ] = mini_batch_preds

        return predictions

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
        
        # Calculate the vector prob
        probabilities = self.predict(X)
        
        ########################################################################
        print(f'evaluate probabilities shape {probabilities.shape}')
        
        # The shape of probabilites after forward propagation is 
        probabilities = probabilities.T
        
        # Compute the accuracy.
        # np.argmax() returns the indices of the maximum values along an axis.
        Y_hat = np.argmax(probabilities, axis=1)
        Y = np.argmax(Y, axis=1)            
        accuracy = (Y_hat == Y).mean()
        accuracy = round(accuracy * 100, 3)               
        print(f'Accuracy: {accuracy}%')  
        
        return