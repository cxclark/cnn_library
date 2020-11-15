import numpy as np
import utils as utils

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
        self.costs = []

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
        
        # Convert labels to binary vector representations of the correct class.
        Y = utils.to_categorical(Y, self.num_classes)
        # Y now has dimesnions (m, num_classes)
                
        # Shuffle X and Y into random mini batches.
        mini_batches = utils.random_mini_batches(X, Y, batch_size)
        
        # Loop through the epochs.
        for epoch in range(epochs):
            print('Running epoch:', epoch + 1)
            
            # Loop through the mini-batches.
            for mini_batch in mini_batches:
                    
                    # Extract mini_batch_X and save it as the input.
                    mini_batch_preds = mini_batch[0]
                    mini_batch_true_labels = mini_batch[1]
                    
                    #Loop through the layers in the network.
                    for layer in self.model:
                        mini_batch_preds = layer.forward(mini_batch_preds)
                    
                    ## DEBUGGING ########################################################
                    print(f'mini_batch[0] shape {mini_batch[0].shape}')
                    print(f'mini_batch_true_labels shape {mini_batch_true_labels.shape}')
                    print(f'mini_batch_preds final shape {mini_batch_preds.shape}')

                    # Compute the loss
                    cost = -np.sum(Y * np.log(mini_batch_preds + 1e-8))
                    print(f'Cost: {cost}')

                    # Compute the accuracy.
                    Y_hat = np.argmax(mini_batch_preds, axis=0)
                    accuracy = (Y_hat == Y).mean()
                    accuracy = accuracy * 100
                    
                    print(f'Cost: {cost}')
                    print(f'Accuracy: {accuracy}')
                    
                    # Compute the derivative.
                    dA = mini_batch_true_labels.T - mini_batch_preds
                    
                    ### DEBUGGING ########################################################
                    print(f'dA.shape: {dA.shape}')
                    
                    # Loop through the reversed layers in the network.
                    for layer in reversed(self.model):
                        dA = layer.backward(dA, learning_rate)


    def predict(self, X):
        """
        Forward propagates through the network and makes predictions on input dataset X.
        Arguments:
            X -- dataset of images, for which to predict labels, numpy array of shape (m, n_H, n_W, c).
        Returns:
            predictions -- label vector, numpy array of shape (m, 10).
        """
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
        predictions = self.predict(X)
        
        predictions = predctions.T
        
        

# def predict(self, X, Y):
# 	A, cache = self.forward(X)
# 	y_hat = np.argmax(A, axis=0)
# 	Y = np.argmax(Y, axis=1)
# 	accuracy = (y_hat == Y).mean()
# 	return accuracy * 100
        
#         predictions = self.predict(X)
        
#         return predictions, Y