import numpy as np

class Perceptron:
    
    def __init__(self, learning_rate=0.01, n_iters=100):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.activation_func = self._unit_step_func
        self.weight = None
        self.bias = None
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weight = np.zeros(n_features)
        self.bias = 0
        
        y_ = np.array([1 if i>0 else 0 for i in y]) 

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weight) + self.bias
                y_predicted = self.activation_func(linear_output)
                
                update = self.lr * (y_predicted - y_[idx])
                self.weight -= update * x_i
                self.bias -= update
                
        
    def predict(self, X):
        linear_output = np.dot(X, self.weight) + self.bias
        y_predicted = self.activation_func(linear_output)
        return y_predicted
        
    def _unit_step_func(self, x):
        return np.where(x>=0, 1, 0)
    
    
