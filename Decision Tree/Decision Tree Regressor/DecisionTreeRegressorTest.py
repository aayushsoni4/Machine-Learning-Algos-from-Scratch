import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Import the DecisionTreeRegressor module
from DecisionTreeRegressor import DecisionTreeRegressor


# Function to calculate root mean squared error (RMSE)
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


# Generate a synthetic regression dataset
X, y = make_regression(n_samples=100, n_features=1, noise=0.1, random_state=42)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234
)

# Instantiate the Decision Tree Regressor with a maximum depth of 10
regressor = DecisionTreeRegressor(max_depth=10)

# Fit the regressor to the training data
regressor.fit(X_train, y_train)

# Make predictions on the test data
y_pred = regressor.predict(X_test)

# Calculate the RMSE of the regressor
rmse_val = rmse(y_test, y_pred)

# Print the RMSE
print("Root Mean Squared Error:", rmse_val)
