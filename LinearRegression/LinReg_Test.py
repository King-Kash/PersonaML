import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets
from LinearRegression import LinearRegression

X, Y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1234)


def mse(predictions, y_test):
    mse = np.mean((predictions-y_test)**2)
    return mse

lin_reg = LinearRegression(lr=0.011, n_iters=1000)
lin_reg.fit(X_train, Y_train)
predictions = lin_reg.predict(X_test)
test_mse = mse(predictions, Y_test)
print("MSE:", test_mse)

#lin_reg line is fit to the sample data
y_pred_line = lin_reg.predict(X)
cmap = plt.get_cmap('viridis')
fig = plt.figure(figsize=(8,6))
m1 = plt.scatter(X_train, Y_train, color='b', s=10)
m2 = plt.scatter(X_test, Y_test, color='r', s=10)
plt.plot(X, y_pred_line, color='black', linewidth=2, label='Prediction')
plt.show()