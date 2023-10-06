import numpy as np

class LogisticRegression:
    
    def __init__(self, lr=0.001, n_iters=1000):
        '''
        Initialize the LogisticRegression object.
        
        Parameters:
        - lr: Learning rate for gradient descent (default is 0.001).
        - n_iters: Number of iterations for training (default is 1000).
        '''
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        '''
        Train the logistic regression model on the input data.
        
        Parameters:
        - X: Input features (numpy array).
        - y: Target labels (numpy array).
        '''
        n_sample, n_features = X.shape
        self.weights = np.zeros(n_features)       
        self.bias = 0
        
        for _ in range(self.n_iters):
            # Calculate the linear model (z = w^T * X + b)
            linear_model = np.dot(X, self.weights) + self.bias
            
            # Apply the sigmoid function to get probabilities (a = sigmoid(z))
            y_predicted = self._sigmoid(linear_model)
            
            # Calculate gradients for weights (dw) and bias (db)
            dw = (1/n_sample) * np.dot(X.T, (y_predicted - y))
            db = (1/n_sample) * np.sum(y_predicted - y)
            
            # Update weights and bias using gradient descent
            self.weights -= self.lr * dw
            self.bias -= self.lr * db       
    
    def predict(self, X):
        '''
        Make predictions on new data.
        
        Parameters:
        - X: Input features for prediction (numpy array).
        
        Returns:
        - y_predicted_cls: Predicted class labels (list of 0s and 1s).
        '''
        # Calculate the linear model (z = w^T * X + b)
        linear_model = np.dot(X, self.weights) + self.bias
        
        # Apply the sigmoid function to get probabilities (a = sigmoid(z))
        y_predicted = self._sigmoid(linear_model)
        
        # Convert probability scores to class labels (0 or 1)
        y_predicted_cls = [1 if i >= 0.5 else 0 for i in y_predicted]
        
        return y_predicted_cls
        
    def _sigmoid(self, z):
        '''
        Sigmoid activation function.
        
        Parameters:
        - z: Input value.
        
        Returns:
        - Output of the sigmoid function (a = 1 / (1 + e^(-z))).
        '''
        return 1 / (1 + np.exp(-z))
