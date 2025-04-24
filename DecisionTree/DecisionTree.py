
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


    def predict():
        pass

