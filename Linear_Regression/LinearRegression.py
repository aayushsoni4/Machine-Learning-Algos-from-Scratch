import numpy as np


class LinearRegression:
    def __init__(self, lr=0.001, n_iters=1000):
        """
        Initialize the LinearRegression object.

        Parameters:
        - lr: Learning rate for gradient descent (default is 0.001). A smaller lr converges slowly but accurately.
        - n_iters: Number of iterations for training (default is 1000). Controls the optimization steps during training.
        """
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None  # Initialize weights as None
        self.bias = None  # Initialize bias as None

    def fit(self, X, y):
        """
        Fit the linear regression model to the training data.

        Parameters:
        - X: Input features (numpy array).
        - y: Target values (numpy array).
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)  # Initialize weights as zeros
        self.bias = 0  # Initialize bias as zero

        for _ in range(self.n_iters):
            # Calculate the predicted values (hypothesis)
            y_predicted = np.dot(X, self.weights) + self.bias

            # Calculate the gradients for weights and bias using the cost function (mean squared error)
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # Update weights and bias using gradient descent to minimize the error
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        """
        Make predictions on new data.

        Parameters:
        - X: Input features for prediction (numpy array).

        Returns:
        - y_predicted: Predicted target values (numpy array).
        """
        # Predict the output based on the learned weights and bias
        y_predicted = np.dot(X, self.weights) + self.bias
        return y_predicted
