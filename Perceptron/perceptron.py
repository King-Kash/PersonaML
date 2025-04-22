import numpy as np
import matplotlib.pyplot as plt

class Perceptron:

    def __init__(self, learning_rate=0.01, max_iterations=1000):
        self.lr = learning_rate
        self.n_iters = max_iterations
        self.weights = None
        #The bias should be updated along with the weights when we run the weight 
        # update fuction on each sample in the training data. From the model, the 
        # bias is labeled w0 (in this implimentation it is updated seperately).
        self.bias = None

    #training method inputs are the data and their coresponding lables
    def train(self, X, Y):
        n_samples, n_features = X.shape

        #init weights (can be done randomly, avoid making them 0, dont know a better way to do this but likely a complex way to do this)
        self.weights = np.random.uniform(0, 1, n_features)
        self.bias = np.random.uniform(0, 1)

        #make sure all labels are 0 or 1 (ex of list comprehension syntax)
        y_ = np.array([1 if i > 0 else 0 for i in Y])

        for _ in range(self.n_iters):
            #enumerate function will give the index and the value at that index
            for idx, x_i in enumerate(X):
                #prediction for one sample unlike predict func which finds prediction for all samples at once
                linear_output = np.dot(self.weights, x_i) + self.bias
                y_predicted = self.activation_function(linear_output)
                
                update = self.lr * (y_[idx] - y_predicted)
                delta_w = update * x_i
                self.weights += delta_w
                self.bias += update * 1 #look at the model for clarification. 
                #we treated the bias as one of the weights and thus we had to 
                #give it a coresponding dummy feature of 1.
                
    
    def activation_function(self, weighted_sum):
        #unit step function
        return np.where(weighted_sum > 0, 1, 0)
        '''
            arr = np.array([1, 2, 3, 4, 5])
            new_arr = np.where(arr > 3 (condition), 100 (if true), 0 (if false))
            print(new_arr)  # Output: [  0   0   0 100 100]
        '''
    #this function is not needed for the traning process, it is purely for the test process where
    #we end up using our final weights and bias to define the linear seperator on the test data. 
    def test_predict(self, X):
        linear_ouput = np.dot(X, self.weights) + self.bias
        y_prediction = self.activation_function(linear_ouput)
        #X is array of all samples. Ouput is array and will be 0 or 1 for each sample.
        return y_prediction