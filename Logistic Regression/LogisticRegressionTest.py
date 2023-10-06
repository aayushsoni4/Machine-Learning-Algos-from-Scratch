import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets

# Import your custom LogisticRegression class
from LogisticRegression import LogisticRegression

# Load the Breast Cancer dataset
bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# Define a function to calculate classification accuracy
def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)

# Create a LogisticRegression object with specified hyperparameters
regressor = LogisticRegression(lr=0.0001, n_iters=1000)

# Train the logistic regression model on the training data
regressor.fit(X_train, y_train)

# Make predictions on the test data
prediction = regressor.predict(X_test)

# Calculate and print the classification accuracy
accuracy_percentage = accuracy(y_test, prediction) * 100
print("Logistic Regression classification accuracy: {:.2f}%".format(accuracy_percentage))