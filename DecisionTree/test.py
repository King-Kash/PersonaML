from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
from DecisionTree import DecisionTree

data = datasets.load_breast_cancer()
X, y = data.data, data.target


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234
)

classifier = DecisionTree() #using all defaults
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)

def accuracy(predictions, y_test):
    correct_score = 0
    for idx, y in enumerate(predictions):
        if y == y_test[idx]:
            correct_score+=1
    return correct_score / len(y_test)

acc = accuracy(predictions, y_test)
print("Model Accuracy:", acc)
