import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from NaiveBayes import NaiveBayes  # Import NaiveBayes class from custom module


# Function to calculate classification accuracy
def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy


# Generate synthetic data
X, y = datasets.make_classification(
    n_samples=1000, n_features=10, n_classes=2, random_state=123
)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123
)

# Initialize Naive Bayes classifier
nb = NaiveBayes()

# Train the Naive Bayes classifier on the training data
nb.fit(X_train, y_train)

# Make predictions on the testing data
predictions = nb.predict(X_test)

# Calculate and print classification accuracy
print(
    "Naive Bayes classification accuracy {}%".format(
        100 * accuracy(y_test, predictions)
    )
)

# Generate and plot confusion matrix
conf_matrix = confusion_matrix(y_test, predictions)
plt.imshow(conf_matrix, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()

# Define class labels for ticks on x and y axes
classes = np.unique(y)
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes)
plt.yticks(tick_marks, classes)

# Add annotations to each cell in the confusion matrix
for i in range(len(classes)):
    for j in range(len(classes)):
        plt.text(
            j,
            i,
            str(conf_matrix[i, j]),
            ha="center",
            va="center",
            color="black",
            fontsize=12,
        )

# Add labels for x and y axes
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.show()
