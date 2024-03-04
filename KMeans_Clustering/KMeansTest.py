import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

# Generate synthetic data for clustering
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# Import your custom KMeans class
from KMeans import KMeans

# Create a KMeans object with the desired number of clusters
kmeans = KMeans(K=4, plot_steps=True)

# Fit the KMeans model on the data
kmeans.fit(X)

# Predict cluster labels for the data
predicted_labels = kmeans.predict(X)

# Plot the final clusters and centroids
plt.figure(figsize=(8, 6))
for i in range(kmeans.K):
    cluster_points = X[predicted_labels == i]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {i}")

centroids = kmeans.centroids
plt.scatter(centroids[:, 0], centroids[:, 1], color="k", marker="x", label="Centroids")
plt.title("KMeans Clustering")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()
