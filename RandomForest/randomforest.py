

class RandomForest:
    def __init__(self, n_trees=10, max_depth=10, min_samples=2, n_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples
        self.n_features = n_features
        self.trees = []
