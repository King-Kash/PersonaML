import numpy as np
from collections import Counter

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None,*,value=None):
        '''*,value=None means we have to pass something into value by name. Only leaf nodes will have values set to something.'''
        self.feature = feature #feature we are going split on
        self.threshold = threshold #threshold for split
        self.left = left #left child
        self.right = right #right child 
        self.value = value #only the leaf nodes will have values

    def is_leaf_node(self): 
        return self.value is not None
        

class DecisionTree:
    def __init__(self, min_sample_split=2, max_depth=100, n_features=None):
        #stoping conditions
        self.min_sample_split=min_sample_split
        self.max_depth=max_depth

        '''
         - Decide how many features do we want to consider for spliting on, store in n_features.
         - Adds randomness since we dont use all features.
         - Important for random forest.
        '''
        self.n_features=n_features
        self.root=None #need access to root node for infrence traversal
    
    def fit(self, X, y):
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1],self.n_features)
        self.root = self._grow_tree(X, y)
    
    def _grow_tree(self, X, y, depth=0):
        #depth keeps track current tree depth
        n_samples, n_feats = X.shape
        n_lables = len(np.unique(y))

        #check stopping criteria:
        if (depth>=self.max_depth or n_samples<self.min_sample_split or n_lables==1):
            '''
             - n_labels == 1 means node has only one label -> its a leaf node do not split it
             - if a given node has less than the min number of samples required to split. -> do not split any further to prevent overfitting
             - if depth of tree exceeds max_depth we decide -> do not split any further
            '''
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)
        
        #pick some of the features out of all the available features without replacement. how many = self.n_features. done for randomness
        feat_idxs = np.random.choice(n_feats, self.n_features, replace=False)

        #find the best split.
        '''Find best feature and then since the feature is numerical the best value of that feature (the threshold) to split the data on'''
        best_feature, best_thresh = self._best_split(X, y, feat_idxs)

        #create children 
        left_idxs, right_idxs = self._split(X[:,best_feature], best_thresh) #split based on best_features and best_thresh
        left = self._grow_tree(X[left_idxs,:], y[left_idxs], depth+1) #define left branch (meaning we say what samples go to left)
        right = self._grow_tree(X[right_idxs,:], y[right_idxs], depth+1) #define right branch (meaning we say what samples go to right)
        return Node(best_feature, best_thresh, left, right) #We formalize the current node that we are at right now. 
        '''Not generating left or right node above. In this whole program we only generate a node when we are at that node.'''



    def _most_common_label(self, y):
        counter = Counter(y)
        value = counter.most_common(1)[0][0]
        return value
    
    def _best_split(self, X, y, feat_idxs):
        best_information_gain = -1
        split_feature_idx, split_threshold = None, None
        
        '''
         -  Two hyper params the feature we want to split on and since that feature is numerical, the threshold we want to split on.
            This function tries every possible combination of the two hyperparameters finding the best combination for maximizing 
            the information gain.
        '''
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
        
        return split_feature_idx, split_threshold

    def _information_gain(self, y, X_column, threshold):
        #parent entropy
        parent_entropy = self._entropy(y)

        #create children
        left_idxs, right_idxs = self._split(X_column, threshold)

        '''
        If all the data goes into one node we have not done any split and uncertainty in the parent and child is identical cuz they contain the same data.
        Weighted Entropy=(0/N * 0) + (N/N * E(parent)) -> IG = E(parent) - E(parent) = 0
        '''
        if len(left_idxs)==0 or len(right_idxs)==0:
            return 0

        #calculate wegihted avg. of entropy of children

        '''following lines to make sure y arrays are 1 dimensional'''
        # y_l = np.asarray(y[left_idxs]).ravel().astype(int)
        # y_r = np.asarray(y[right_idxs]).ravel().astype(int)

        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        weighted_child_entropy = (n_l/n)*e_l + (n_r/n)*e_r

        #calculate information gain
        information_gain = parent_entropy - weighted_child_entropy
        return information_gain
    

    #entropy of node. This is not conditional entropy caculation.
    def _entropy(self, y):
        '''Bincount creates a histogram of relative frequency for each element that appears in y'''
        # y = np.array(y).ravel().astype(int) //makes y one dimensional array
        histogram = np.bincount(y)
        probabilities = histogram / len(y) #calculate probability of each possible y occuring. all of them stored in one array.
        return -np.sum([prob * np.log(prob) for prob in probabilities if prob>0]) #sum up all the probabilities
    
    #split the parent entries into two child entries based on threshold
    def _split(self, X_column, split_threshold):
        '''Argwhere will return new array (N,1) with entries less than threshold. Flatten to make (N,) dim array.'''
        left_idxs = np.argwhere(X_column<=split_threshold).flatten()
        right_idx = np.argwhere(X_column>split_threshold).flatten()
        return left_idxs, right_idx
        

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])
    
    def _traverse_tree(self, x, node):
        #eventualy every traversal ends in a leaf node
        if node.is_leaf_node():
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)
    





