import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

# Generate synthetic data for regression
X, y = datasets.make_regression(n_samples=100, n_features=1, noise=10)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Import your custom LinearRegression class
from LinearRegression import LinearRegression

# Create a LinearRegression object with a specified learning rate
LR = LinearRegression(lr=0.01)

# Fit the linear regression model on the training data
LR.fit(X_train, y_train)

# Predict the target values on the test set
predicted = LR.predict(X_test)

# Define a function to calculate Mean Squared Error (MSE)
def mse(y_test, predicted):
    return np.mean((y_test - predicted) ** 2)

# Calculate and print the MSE between predicted and actual values
mse_value = mse(y_test, predicted)
print(mse_value)

# Predict values for the entire dataset
y_pred_line = LR.predict(X)

# Create a scatter plot to visualize the data points
cmap = plt.get_cmap('viridis')
fig = plt.figure(figsize=(8, 6))
m1 = plt.scatter(X_train, y_train, color=cmap(0.9), s=10, label='Training Data')
m2 = plt.scatter(X_test, y_test, color=cmap(0.5), s=10, label='Testing Data')

# Plot the linear regression line
plt.plot(X, y_pred_line, color='black', linewidth=2, label='Prediction')

# Add labels and legend to the plot
plt.xlabel('X')
plt.ylabel('y')
plt.legend(loc='upper left')

# Display the plot
plt.show()
