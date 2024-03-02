import numpy as np
import matplotlib.pyplot as plt

class KMeans:
    """
    KMeans clustering algorithm.

    Parameters:
    - K: Number of clusters (default is 5).
    - max_iters: Maximum number of iterations for convergence (default is 100).
    - plot_steps: Whether to plot the clusters at each iteration (default is False).
    - n_init: Number of times the k-means algorithm is run with different centroid seeds (default is 10).
    """

    def __init__(self, K=5, max_iters=100, plot_steps=False, n_init=10):
        """
        Initialize the KMeans object.

        Parameters:
        - K: Number of clusters (default is 5).
        - max_iters: Maximum number of iterations for convergence (default is 100).
        - plot_steps: Whether to plot the clusters at each iteration (default is False).
        - n_init: Number of times the k-means algorithm is run with different centroid seeds (default is 10).
        """
        self.K = K 
        self.max_iters = max_iters
        self.plot_steps = plot_steps
        self.n_init = n_init

        self.clusters = [[] for _ in range(self.K)]
        self.centroids = []
  
    def fit(self, X):
        """
        Fit the KMeans clustering model to the data.

        Parameters:
        - X: Input data (numpy array).
        """
        self.X = X
        self.n_samples, self.n_features = X.shape
        
        # Initialize best centroids and minimum inertia
        best_centroids = None
        min_inertia = float('inf')

        for _ in range(self.n_init):
            # Step 1: Initialize centroids randomly
            random_indices = np.random.choice(self.n_samples, size=self.K, replace=False)
            centroids = [self.X[idx] for idx in random_indices]

            for _ in range(self.max_iters):
                # Step 2: Assign data points to clusters
                clusters = self._assign_clusters(centroids)
                
                # Step 3: Update centroids
                prev_centroids = centroids
                centroids = self._update_centroids(clusters)
                
                # Step 4: Convergence check
                if self._is_converged(prev_centroids, centroids):
                    break

            # Calculate inertia
            inertia = self._calculate_inertia(clusters, centroids)
            
            # Update best centroids and minimum inertia if current inertia is lower
            if inertia < min_inertia:
                min_inertia = inertia
                best_centroids = centroids.copy()
            
        # Set the best centroids found
        self.centroids = best_centroids

        # Assign data points to final clusters
        self.clusters = self._assign_clusters(self.centroids)
        
        # Plotting (optional)
        if self.plot_steps:
            self.plot()
    
    def _assign_clusters(self, centroids):
        """
        Assign each data point to the nearest centroid.

        Parameters:
        - centroids: List of centroids.

        Returns:
        - clusters: List of clusters, where each cluster is a list of data point indices.
        """
        clusters = [[] for _ in range(self.K)]
        for idx, sample in enumerate(self.X):
            distances = [np.linalg.norm(sample - centroid) for centroid in centroids]
            cluster_idx = np.argmin(distances)
            clusters[cluster_idx].append(idx)
        return clusters
    
    def _update_centroids(self, clusters):
        """
        Update centroids based on the mean of data points in each cluster.

        Parameters:
        - clusters: List of clusters, where each cluster is a list of data point indices.

        Returns:
        - centroids: Updated centroids.
        """
        centroids = np.zeros((self.K, self.n_features))
        for cluster_idx, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.X[cluster], axis=0)
            centroids[cluster_idx] = cluster_mean
        return centroids
    
    def _is_converged(self, prev_centroids, centroids, tol=1e-6):
        """
        Check for convergence by comparing the new centroids with the previous centroids.

        Parameters:
        - prev_centroids: Previous centroids.
        - centroids: Current centroids.
        - tol: Tolerance for convergence (default is 1e-6).

        Returns:
        - converged: True if centroids have converged, False otherwise.
        """
        return all(np.linalg.norm(prev_centroid - centroid) < tol for prev_centroid, centroid in zip(prev_centroids, centroids))
    
    def _calculate_inertia(self, clusters, centroids):
        """
        Calculate the inertia (total within-cluster sum of squares).

        Parameters:
        - clusters: List of clusters, where each cluster is a list of data point indices.
        - centroids: List of centroids.

        Returns:
        - inertia: Total within-cluster sum of squares.
        """
        inertia = 0
        for cluster_idx, cluster in enumerate(clusters):
            cluster_inertia = np.sum(np.linalg.norm(self.X[cluster] - centroids[cluster_idx], axis=1) ** 2)
            inertia += cluster_inertia
        return inertia
    
    def predict(self, X):
        """
        Predict cluster labels for new data points.

        Parameters:
        - X: New data points (numpy array).

        Returns:
        - labels: Predicted cluster labels.
        """
        distances = np.linalg.norm(X[:, np.newaxis, :] - self.centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        return labels
    
    def plot(self):
        """
        Plot the clusters and centroids.
        """
        plt.figure(figsize=(12, 8))
        for cluster_idx, cluster in enumerate(self.clusters):
            cluster_points = self.X[cluster]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster_idx}')
        plt.scatter(np.array(self.centroids)[:, 0], np.array(self.centroids)[:, 1], color='k', marker='x', label='Centroids')
        plt.title('KMeans Clustering')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.show()
