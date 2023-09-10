import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

X, y = datasets.make_regression(n_samples=100, n_features=1, noise=10)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


from LinearRegression import LinearRegression

LR = LinearRegression(lr=0.01)
LR.fit(X_train, y_train)

predicted = LR.predict(X_test)

def mse(y_test, predicted):
    return np.mean((y_test-predicted)**2)

mse_value = mse(y_test, predicted)
print(mse_value)

y_pred_line = LR.predict(X)
cmap = plt.get_cmap('viridis')
fig = plt.figure(figsize=(8,6))
m1 = plt.scatter(X_train, y_train, color=cmap(0.9), s=10)
m2 = plt.scatter(X_test, y_test, color=cmap(0.5), s=10)
plt.plot(X, y_pred_line, color='black', linewidth=2, label='Prediction')
plt.show()
