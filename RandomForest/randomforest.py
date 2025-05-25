from PersonaML.DecisionTree.DecisionTree import DecisionTree
import numpy as np
from collections import Counter

class RandomForest:
    def __init__(self, n_trees=10, max_depth=10, min_samples=2, n_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples
        self.n_features = n_features
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTree(max_depth=self.max_depth,
                                min_sample_split=self.min_samples_split, 
                                n_features=self.n_features)
            X_sample, y_sample = self._bootstrap_sampling(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
    
    #bagging
    def _bootstrap_sampling(self, X, y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]
    
    #majority vote
    def _most_common_label(self, y):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common
    
    def predict(self, X):
        for tree in self.trees:
            #list comprehension get used to it.
            predictions = np.array([tree.predict(X) for tree in self.trees])
            '''[ [1, 0, 1, 1],
                 [0, 0, 1, 0], ...] #each tree produces a list of predictions
            # we need to reformat this array to be in this form
                #[ [predictions for sample one from each tree], [], [] ]'''
            #this takes the cols which correspond to the prediction for a particular sample by all the trees and makes it a row.
            trees_preds = np.swapaxes(predictions, 0, 1)
            return np.array([self._most_common_label(pred) for pred in trees_preds])


            