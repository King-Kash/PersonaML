import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self, lr=0.01, n_iters = 1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
    
    #Fit function if for training
    def fit(self, X, Y):
        #num of rows and cols
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            #unlike perceptron we want to do prediction for all samples at once
            #y = wX + b -> dot product of (X * w) + b
            y_pred = np.dot(X, self.weights) + self.bias

        #__________numpy implimentation of average error and update step__________
            dw = (1.0/n_samples)*2*np.dot(X.T,(y_pred - Y)) #ex. if X nxn and Y&y_pred nx1 then the dot product of the two will also be nx1. that is some scalar to subtract from each current weight.
            db = (1.0/n_samples)*2*np.sum((y_pred - Y))
            self.weights -= self.lr*dw
            self.bias -= self.lr*db

        #__________manual implimentation of the average error and update step__________
            # update_sum_dw = 0
            # update_sum_db = 0
            # for idx, x_i in enumerate(X):
            #     #note that the Y_pred and the label or a single scalar value. This is the behavior of regression.
            #     update_sum_dw += 2*x_i*(y_pred[idx] - Y[idx])
            #     update_sum_db += 2*(y_pred[idx] - Y[idx])
            # self.weights -= self.lr*(1.0/n_samples)*update_sum_dw
            # self.bias -= self.lr*(1.0/n_samples)*update_sum_db


    #Predict function for infrence
    def predict(self, X):
        #y = mx+b
        y_pred = np.dot(X, self.weights) + self.bias
        return y_pred
    
#Summary:

'''
prediction => y_pred = wx+b
update => w_new = w_old + (lr)(dw)
          bias_new = bias_old + (lr)(db)

Goal minimize the loss function (Mean square error) by finding the optimal weights and biases.
N - number of training samples
MSE = J(w,b) = 1/N * np.sum((y_pred - y)**2)
y_pred = wx+b -> W.T*X + b (in vector form wher capitals indicate vectors and .T means transpose)
J(w,b) = 1/N * np.sum(((wx+b) - y)**2)
dJ/dw = dw = 1/N * np.sum(2((y_pred=wx+b) - y)(x))
dJ/db = db = 1/N * np.sum(2((y_pred=wx+b) - y))

Many of the calculations in the code are performed on all samples at once for efficency. Please pay attention
to this fact. Above I have implimented the update step in two ways, 1. multiply each sample of X with its re-
pective error all at once then find dw 2. Use running sum to add up sample * error, then calculate average at
end.


'''



    