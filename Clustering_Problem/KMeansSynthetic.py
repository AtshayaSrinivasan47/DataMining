# Atshaya Srinivasan- 201774445

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings

# Question-2 (clustering K means using synthetically generated data points as provided size of dataset)

# Set a fixed seed for reproducibility
SEED = 42
np.random.seed(SEED)

# Function to generate synthetic data
def generate_synthetic_data(size, dim):
    """
    Generates synthetic data.

    Parameters:
    - size: Number of data points to generate.
    - dim: Dimensionality of each data point.

    Returns:
    - Synthetic data array of shape (size, dim).
    """
    return np.random.rand(size, dim)

# Function to calculate Euclidean distance
def ComputeDistance(x1, x2):
    """
    Calculates the Euclidean distance between two points.

    Parameters:
    - x1: First point.
    - x2: Second point.

    Returns:
    - Euclidean distance between x1 and x2.
    """
    return np.sqrt(np.sum((x1 - x2) ** 2))

# Function to initialize centroids randomly
def initialSelection(X, k):
    """Initialize centroids randomly.

    Args:
        X (numpy.ndarray): Data points.
        k (int): Number of clusters.

    Returns:
        numpy.ndarray: Initialized centroids.
    """
    indices = np.random.choice(X.shape[0], k, replace=False)
    centroids = X[indices]
    return centroids

# Function to assign cluster ids to each data point
def assignClusterIds(X, centroids):
    """
    Assigns cluster ids to each data point.

    Parameters:
    - X: Data array of shape (n_samples, n_features).
    - centroids: Array of centroids with shape (k, n_features).

    Returns:
    - Array of cluster ids for each data point.
    """
    labels = []
    for x in X:
        distances = [ComputeDistance(x, c) for c in centroids]
        labels.append(np.argmin(distances))
    return np.array(labels)

# Function to compute cluster representatives
def computeClusterRepresentatives(X, labels, k):
    """
    Computes cluster representatives.

    Parameters:
    - X: Data array of shape (n_samples, n_features).
    - labels: Array of cluster ids for each data point.
    - k: Number of clusters.

    Returns:
    - Array of centroids representing each cluster.
    """
    centroids = []
    for i in range(k):
        cluster_i = X[labels == i]
        if cluster_i.size == 0:
            centroids.append(np.zeros_like(X[0]))  # Add zeros as representatives for empty clusters
        else:
            centroid_i = np.mean(cluster_i, axis=0)
            centroids.append(centroid_i)
    return np.array(centroids)

# Function to compute the Silhouette coefficient
def compute_silhouette_coefficient(X, labels, k):
    """
    Computes the Silhouette coefficient for a set of clusters.

    Parameters:
    - X: Data array of shape (n_samples, n_features).
    - labels: Array of cluster ids for each data point.
    - k: Number of clusters.

    Returns:
    - Silhouette coefficient.
    """
    n = X.shape[0]
    silhouette_coefficients = []

    for i in range(n):
        cluster_i = X[labels == labels[i]]

        # Skip data points belonging to empty clusters
        if cluster_i.size == 0:
            continue

        # Compute a_i (mean intra-cluster distance)
        a_i = np.mean([ComputeDistance(X[i], c) for c in cluster_i if not np.array_equal(X[i], c)])

        # Compute b_i (mean nearest-cluster distance)
        b_i = np.inf
        for j in range(k):
            if j != labels[i]:
                cluster_j = X[labels == j]
                if cluster_j.size > 0:
                    b_i = min(b_i, np.mean([ComputeDistance(X[i], c) for c in cluster_j]))

        # Add a small constant to avoid division by zero
        a_i += 1e-10
        b_i += 1e-10

        # Compute Silhouette coefficient
        if a_i == 0 or b_i == 0:
            s_i = 0
        else:
            s_i = (b_i - a_i) / max(a_i, b_i)

        silhouette_coefficients.append(s_i)

    # If all data points belong to empty clusters, return 0
    if not silhouette_coefficients:
        return 0.0

    return np.mean(silhouette_coefficients)

# Function to perform k-means clustering
def kmeans(X, k, maxIter=100):
    """
    Performs k-means clustering.

    Parameters:
    - X: Data array of shape (n_samples, n_features).
    - k: Number of clusters.
    - maxIter: Maximum number of iterations.

    Returns:
    - Array of cluster ids for each data point.
    """
    centroids = initialSelection(X, k)

    for _ in range(maxIter):
        labels = assignClusterIds(X, centroids)
        new_centroids = computeClusterRepresentatives(X, labels, k)

        if np.allclose(centroids, new_centroids):
            break

        centroids = new_centroids

    return labels

# Function to plot Silhouette coefficients
def plot_silhouette(silhouette_coefficients, k):
    """
    Plots the Silhouette coefficients for different values of k.

    Parameters:
    - silhouette_coefficients: List of Silhouette coefficients.
    - k: List of values of k.

    Returns:
    - None
    """
    plt.figure(figsize=(8, 6))
    plt.plot(k, silhouette_coefficients, marker='o')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Coefficient')
    plt.title('Silhouette Coefficient vs clusters - Question 2')
    plt.grid(True)
    plt.xticks(k)
    plt.savefig('Question2_Silhouette_Plot.png') # saving png file in current directory
    plt.show()
    plt.close()

# Load the provided dataset to get the size
data = pd.read_csv('dataset', header=None, sep='\s+')
size = len(data)

# Filter out the runtime warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Generate synthetic data of the same size
dim = data.shape[1] - 1
synthetic_data = generate_synthetic_data(size, dim)

# Compute Silhouette coefficients for different values of k
silhouette_coefficients = []
k = range(1, 10)

for ks in k:
    labels = kmeans(synthetic_data, ks)
    silhouette_coefficients.append(compute_silhouette_coefficient(synthetic_data, labels, ks))


# Plot the results and save the plot in the current folder
plot_silhouette(silhouette_coefficients, k)

"""The consistently low Silhouette Coefficient values across various cluster numbers indicate potential
unsuitability of the data for clustering or inadequacy of the chosen clustering algorithm and its
parameters for this dataset. It's plausible that the data lacks clear, distinguishable clusters, 
or the selected features for clustering might not sufficiently encapsulate the underlying structure."""