import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

# Importing the DecisionTreeClassifier module
from DecisionTreeClassifier import DecisionTreeClassifier


# Function to calculate accuracy
def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy


# Load the Breast Cancer dataset
data = datasets.load_breast_cancer()
X = data.data
y = data.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234
)

# Instantiate the Decision Tree Classifier with a maximum depth of 10
clf = DecisionTreeClassifier(max_depth=10)

# Fit the classifier to the training data
clf.fit(X_train, y_train)

# Make predictions on the test data
y_pred = clf.predict(X_test)

# Calculate the accuracy of the classifier
acc = accuracy(y_test, y_pred)

# Print the accuracy
print("Accuracy:", acc)
