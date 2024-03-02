import numpy as np
from collections import Counter
from DecisionTreeClassifier import DecisionTreeClassifier


class RandomForestClassifier:
    """
    Random Forest Classifier.

    Parameters:
    - n_trees: The number of decision trees in the forest.
    - min_samples_split: The minimum number of samples required to split an internal node in each decision tree (default is 2).
    - max_depth: The maximum depth of each decision tree (default is 100).
    - n_feats: The number of features to consider when looking for the best split in each decision tree (default is None).
    """

    def __init__(self, n_trees=100, min_samples_split=2, max_depth=100, n_feats=None):
        """
        Initialize the Random Forest Classifier.

        Parameters:
        - n_trees: The number of decision trees in the forest.
        - min_samples_split: The minimum number of samples required to split an internal node in each decision tree (default is 2).
        - max_depth: The maximum depth of each decision tree (default is 100).
        - n_feats: The number of features to consider when looking for the best split in each decision tree (default is None).
        """
        self.n_trees = n_trees
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.trees = []

    def fit(self, X, y):
        """
        Fit the random forest classifier to the training data.

        Parameters:
        - X: Input features (numpy array).
        - y: Target labels (numpy array).
        """
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTreeClassifier(
                min_samples_split=self.min_samples_split,
                max_depth=self.max_depth,
                n_feats=self.n_feats,
            )
            X_sample, y_sample = self._bootstrap_sample(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        """
        Make predictions using the trained random forest classifier.

        Parameters:
        - X: Input features for prediction (numpy array).

        Returns:
        - predictions: Predicted labels (numpy array).
        """
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(tree_preds, 0, 1)

        y_pred = [
            self._most_common_label(tree_preds[i]) for i in range(tree_preds.shape[0])
        ]
        return np.array(y_pred)

    def _most_common_label(self, y):
        """
        Find the most common label in a set of labels.

        Parameters:
        - y: Target labels (numpy array).

        Returns:
        - most_common: Most common label.
        """
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common

    def _bootstrap_sample(self, X, y):
        """
        Create a bootstrap sample of the data.

        Parameters:
        - X: Input features (numpy array).
        - y: Target labels (numpy array).

        Returns:
        - X_sample: Bootstrapped input features (numpy array).
        - y_sample: Bootstrapped target labels (numpy array).
        """
        n_samples = X.shape[0]
        indices = np.random.randint(n_samples, size=n_samples, replace=True)
        return X[indices], y[indices]
