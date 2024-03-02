import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

from Perceptron import Perceptron  # Import the Perceptron class


# Function to calculate accuracy
def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy


# Generate synthetic data
X, y = datasets.make_blobs(
    n_samples=150, n_features=2, centers=2, cluster_std=1.05, random_state=2
)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123
)

# Initialize and train the Perceptron model
p = Perceptron(learning_rate=0.01, n_iters=1000)
p.fit(X_train, y_train)

# Make predictions on the test set
predictions = p.predict(X_test)

# Calculate and print the classification accuracy
print("Perceptron classification accuracy:", accuracy(y_test, predictions))

# Plot the training data points
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plt.scatter(X_train[:, 0], X_train[:, 1], marker="o", c=y_train)

# Plot the decision boundary learned by the Perceptron
x0_1 = np.amin(X_train[:, 0])
x0_2 = np.amax(X_train[:, 0])
x1_1 = (-p.weight[0] * x0_1 - p.bias) / p.weight[1]
x1_2 = (-p.weight[0] * x0_2 - p.bias) / p.weight[1]

ax.plot([x0_1, x0_2], [x1_1, x1_2], "k")

# Set y-axis limits for better visualization
ymin = np.amin(X_train[:, 1])
ymax = np.amax(X_train[:, 1])
ax.set_ylim([ymin - 3, ymax + 3])

# Show the plot
plt.show()
