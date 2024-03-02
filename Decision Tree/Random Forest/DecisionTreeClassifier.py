import numpy as np
from collections import Counter


class Node:
    """
    Node in the decision tree.

    Parameters:
    - features: Index of the feature used for splitting.
    - threshold: Threshold value for the split.
    - left: Left child node.
    - right: Right child node.
    - value: Value of the node for leaf nodes (class label).
    """

    def __init__(
        self, features=None, threshold=None, left=None, right=None, *, value=None
    ):
        """
        Initialize a node in the decision tree.

        Parameters:
        - features: Index of the feature used for splitting.
        - threshold: Threshold value for the split.
        - left: Left child node.
        - right: Right child node.
        - value: Value of the node for leaf nodes (class label).
        """
        self.features = features
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        """
        Check if the node is a leaf node.

        Returns:
        - True if the node is a leaf node, False otherwise.
        """
        return self.value is not None


class DecisionTreeClassifier:
    """
    Decision Tree Classifier.

    Parameters:
    - min_samples_split: The minimum number of samples required to split an internal node (default is 2).
    - max_depth: The maximum depth of the tree (default is 100).
    - n_feats: The number of features to consider when looking for the best split (default is None).
    """

    def __init__(self, min_samples_split=2, max_depth=100, n_feats=None):
        """
        Initialize the Decision Tree Classifier.

        Parameters:
        - min_samples_split: The minimum number of samples required to split an internal node (default is 2).
        - max_depth: The maximum depth of the tree (default is 100).
        - n_feats: The number of features to consider when looking for the best split (default is None).
        """
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.root = None

    def fit(self, X, y):
        """
        Fit the decision tree classifier to the training data.

        Parameters:
        - X: Input features (numpy array).
        - y: Target labels (numpy array).
        """
        self.n_feats = X.shape[1] if not self.n_feats else min(self.n_feats, X.shape[1])
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        """
        Recursively grow the decision tree.

        Parameters:
        - X: Input features (numpy array).
        - y: Target labels (numpy array).
        - depth: Current depth of the tree (default is 0).

        Returns:
        - node: The root node of the decision tree.
        """
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # stopping criteria
        if (
            depth >= self.max_depth
            or n_labels == 1
            or n_samples < self.min_samples_split
        ):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        feat_idxs = np.random.choice(n_features, self.n_feats, replace=False)

        best_feat, best_thres = self._best_criteria(X, y, feat_idxs)

        left_idxs, right_idxs = self._split(X[:, best_feat], best_thres)

        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)

        return Node(best_feat, best_thres, left, right)

    def _best_criteria(self, X, y, feat_idxs):
        """
        Find the best split criteria for the decision tree.

        Parameters:
        - X: Input features (numpy array).
        - y: Target labels (numpy array).
        - feat_idxs: Indices of the features to consider for splitting.

        Returns:
        - split_idx: Index of the feature used for splitting.
        - split_thres: Threshold value for the split.
        """
        best_gain = -1
        split_idx, split_thres = None, None

        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)

            for threshold in thresholds:
                gain = self._information_gain(y, X_column, threshold)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thres = threshold

        return split_idx, split_thres

    def _information_gain(self, y, X_column, split_thres):
        """
        Calculate the information gain for a split.

        Parameters:
        - y: Target labels (numpy array).
        - X_column: Feature column used for the split.
        - split_thres: Threshold value for the split.

        Returns:
        - ig: Information gain value.
        """
        parent_entropy = self._entropy(y)

        left_idxs, right_idxs = self._split(X_column, split_thres)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])

        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r

        ig = parent_entropy - child_entropy
        return ig

    def _split(self, X_column, split_thres):
        """
        Split the data based on a threshold.

        Parameters:
        - X_column: Feature column used for the split.
        - split_thres: Threshold value for the split.

        Returns:
        - left_idxs: Indices of the samples that satisfy the split condition.
        - right_idxs: Indices of the samples that do not satisfy the split condition.
        """
        left_idxs = np.argwhere(X_column <= split_thres).flatten()
        right_idxs = np.argwhere(X_column > split_thres).flatten()
        return left_idxs, right_idxs

    def _entropy(self, y):
        """
        Calculate the entropy of a set of labels.

        Parameters:
        - y: Target labels (numpy array).

        Returns:
        - entropy: Entropy value.
        """
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log2(p) for p in ps if p > 0])

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

    def predict(self, X):
        """
        Make predictions using the trained decision tree.

        Parameters:
        - X: Input features for prediction (numpy array).

        Returns:
        - predictions: Predicted labels (numpy array).
        """
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        """
        Traverse the decision tree to make predictions for a single data point.

        Parameters:
        - x: Input features for a single data point.
        - node: Current node in the decision tree.

        Returns:
        - predicted_label: Predicted label for the data point.
        """
        if node.is_leaf_node():
            return node.value

        if x[node.features] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)
