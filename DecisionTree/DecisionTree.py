import numpy as np
from collections import Counter

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None,*,value=None):
        #*,value=None means we have to pass something into value by name. Only leaf nodes will have values set to something.
        self.feature = feature #feature we are going split on
        self.threshold = threshold #threshold for split
        self.left = left #left child
        self.right = right #right child 
        self.value = None #only the leaf nodes will have values

        def is_leaf_node(self): 
            return self.value is not None
        

class DecisionTree:
    def __init__(self, min_sample_split=2, max_depth=100, n_features=None):
        #stoping conditions
        self.min_sample_split=min_sample_split
        self.max_depth=max_depth
        #Decide how many features do we want to split on
        #Adds randomness since we dont use all features
        #Important for random forest
        self.n_features=n_features
        #need access to root node for infrence traversal
        self.root=None
    
    def fit(self, X, y):
        if self.n_features is None:
            self.n_features = X.shape[1]
        else:
            self.n_features = min(X.shape[1], self.n_features)

        self.root = self.grow_tree(X, y)
    
    def _grow_tree(self, X, y, depth=0):
        #depth keeps track current tree depth
        n_samples, n_feats = X.shape
        n_lables = len(np.unique(y))

        #check stopping criteria:
        if (depth>=self.max_depth or n_samples<self.min_sample_split or n_lables==1):
            #n_labels == 1 means node has only one label -> its a leaf node do not split it
            #if a given node has less than the min number of samples required to split. -> do not split any further to prevent overfitting
            #if depth of tree exceeds max_depth we decide -> do not split any further
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)
        
        #pick some of the features out of all the available features without replacement. how many = self.n_features. done for randomness
        feat_idxs = np.random.choice(n_feats, self.n_features, replace=False)
        #find the best feature and then since the feature is numerical the best value of that feature (the threshold) to split the data on
        best_features, best_thresh = self._best_split(X, y, feat_idxs)

    def _most_common_label(self, y):
        counter = Counter(y)
        value = counter.most_common(1)[0][0]
        return value
    
    def _best_split(self, X, y, feat_idxs):
        best_information_gain = -1
        split_feature_idx, split_threshold = None, None
        
        #2 hyper params the feature we want to split on and since that feature is numerical, the threshold we want to sliton
        #thif function tries every possible combination of the two hyperparameters finding the best combination for maximizing the information gain
        for feat_idx in feat_idxs:
            #one column of X containg the values of the feature at feat_idx for each sample.
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column) #all the options we have for where we want the split to be.

            for thr in thresholds:
                #calculate information gain for current feature, threshold combination
                gain = self._information_gain(y, X_column, thr)
                if gain > best_information_gain:
                    best_information_gain = gain
                    split_feature_idx = feat_idx
                    split_threshold = thr

    def _information_gain(self, y, X_column, threshold):
        #parent entropy
        parent_entropy = self._entropy(y)

        #create children

        #calculate wegihted avg. of entropy of children

        #calculate information gain
    
    def _entropy(self, y):
        #bincount creates a histogram of relative frequency for each element that appears in y
        histogram = np.bincount(y)
        probabilities = histogram / len(y) #calculate probability of each possible y occuring all at once and store in array
        return -np.sum([prob * np.log(prob) for prob in probabilities if prob>0]) #sum up all the probabilities


    def predict():
        pass




