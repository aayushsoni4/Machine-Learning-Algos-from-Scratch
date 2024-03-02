import numpy as np


class NaiveBayes:
    """
    Naive Bayes classifier implementation.

    Attributes:
    - _mean: Mean of each feature for each class.
    - _var: Variance of each feature for each class.
    - _priors: Prior probabilities of each class.
    - _classes: Unique classes in the training data.
    """

    def __init__(self):
        """
        Initialize the NaiveBayes object.
        """
        self._mean = None
        self._var = None
        self._priors = None
        self._classes = None

    def fit(self, X, y):
        """
        Train the Naive Bayes model on the input data.

        Parameters:
        - X: Input features (numpy array).
        - y: Target labels (numpy array).
        """
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)

        # Initialize arrays to store mean, variance, and priors
        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._var = np.zeros((n_classes, n_features), dtype=np.float64)
        self._priors = np.zeros(n_classes, dtype=np.float64)

        # Calculate mean, variance, and priors for each class
        for c in self._classes:
            X_c = X[c == y]
            self._mean[c, :] = X_c.mean(axis=0)
            self._var[c, :] = X_c.var(axis=0)
            self._priors[c] = X_c.shape[0] / float(n_samples)

    def predict(self, X):
        """
        Make predictions on new data.

        Parameters:
        - X: Input features for prediction (numpy array).

        Returns:
        - y_pred: Predicted class labels (list of labels).
        """
        # Calculate the posterior probabilities for each class
        y_pred = [self._predict(x) for x in X]
        return y_pred

    def _predict(self, x):
        """
        Predict the class for a single data point.

        Parameters:
        - x: Input features for a single data point.

        Returns:
        - Predicted class label.
        """
        posteriors = []

        # Calculate the log posterior probability for each class
        for idx, c in enumerate(self._classes):
            prior = np.log(self._priors[idx])
            class_conditional = np.sum(np.log(self._pdf(idx, x)))
            posterior = prior + class_conditional
            posteriors.append(posterior)

        # Return the class with the highest log posterior probability
        return self._classes[np.argmax(posteriors)]

    def _pdf(self, class_idx, x):
        """
        Calculate the probability density function for a given class and data point.

        Parameters:
        - class_idx: Index of the class.
        - x: Input features for a single data point.

        Returns:
        - Probability density function value.
        """
        mean = self._mean[class_idx]
        var = self._var[class_idx]
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator
