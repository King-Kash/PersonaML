import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

from perceptron import Perceptron

def accuracy(y_true, y_pred):
    accuracy = round(np.sum(y_true == y_pred) / (len(y_pred) * 1.0), 2)
    return accuracy

#we can use sklearn to generate our data (notice the importance of correcting the labels in the perceptron train method)
#if the standard deviation is too high then it will be very unlikely the data will be linearly seperable.
X, y = datasets.make_blobs(n_samples=150, n_features=2, centers=2, cluster_std=1.05, random_state=2) #autmoaticall generates data & their respective labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

p = Perceptron(learning_rate=0.01, max_iterations=1000)
p.train(X_train, y_train)
predictions = p.test_predict(X_test)
model_accuracy = accuracy(y_test, predictions)

print("Perceptron classification accuracy: ", model_accuracy)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plt.scatter(X_train[:,0],X_train[:,1], marker='o', c=y_train, s=2)

x0_1 = np.amin(X_train[:,0])
x0_2 = np.amax(X_train[:,0])

if np.isclose(p.weights[1], 0):
    print("Warning: p.weights[1] is too close to zero; cannot plot a valid decision boundary.")
else:
    x1_1 = (-p.weights[0] * x0_1 - p.bias) / p.weights[1]
    x1_2 = (-p.weights[0] * x0_2 - p.bias) / p.weights[1]
    ax.plot([x0_1, x0_2], [x1_1, x1_2], 'k')


ymin = np.amin(X_train[:, 1])
ymax = np.amax(X_train[:, 1])
ax.set_ylim([ymin-3, ymax+3])

plt.show()