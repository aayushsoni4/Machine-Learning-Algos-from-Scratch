import numpy as np


class Perceptron:
    """
    Perceptron classifier.

    Parameters:
    - learning_rate: Learning rate for updating the weights (default is 0.01).
    - n_iters: Number of iterations for training (default is 100).
    """

    def __init__(self, learning_rate=0.01, n_iters=100):
        """
        Initialize the Perceptron object.

        Parameters:
        - learning_rate: Learning rate for updating the weights (default is 0.01).
        - n_iters: Number of iterations for training (default is 100).
        """
        self.lr = learning_rate
        self.n_iters = n_iters
        self.activation_func = self._unit_step_func
        self.weight = None
        self.bias = None

    def fit(self, X, y):
        """
        Fit the Perceptron model to the training data.

        Parameters:
        - X: Input features (numpy array).
        - y: Target values (numpy array).
        """
        n_samples, n_features = X.shape
        self.weight = np.zeros(n_features)
        self.bias = 0

        # Convert target values to binary (0 or 1)
        y_ = np.array([1 if i > 0 else 0 for i in y])

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weight) + self.bias
                y_predicted = self.activation_func(linear_output)

                update = self.lr * (y_predicted - y_[idx])
                self.weight -= update * x_i
                self.bias -= update

    def predict(self, X):
        """
        Make predictions on new data.

        Parameters:
        - X: Input features for prediction (numpy array).

        Returns:
        - y_predicted: Predicted target values (numpy array).
        """
        linear_output = np.dot(X, self.weight) + self.bias
        y_predicted = self.activation_func(linear_output)
        return y_predicted

    def _unit_step_func(self, x):
        """
        Step function for activation.

        Parameters:
        - x: Input value.

        Returns:
        - Output value after activation.
        """
        return np.where(x >= 0, 1, 0)
