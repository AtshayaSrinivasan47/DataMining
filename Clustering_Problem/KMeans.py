"""Atshaya Srinivasan - 201774445"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

""" Question 1 - The code implements the k-means clustering algorithm and calculates the Silhouette coefficient,
which is a metric to evaluate the quality of the clustering. """


def ComputeDistance(x1, x2):
    """Calculate Euclidean distance between two points.

    Args:
        x1 (numpy.ndarray): Coordinates of the first point.
        x2 (numpy.ndarray): Coordinates of the second point.

    Returns:
        float: Euclidean distance between x1 and x2.
    """
    return np.sqrt(np.sum((x1 - x2) ** 2))


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


def assignClusterIds(X, centroids):
    """Assign cluster IDs to each data point.

    Args:
        X (numpy.ndarray): Data points.
        centroids (numpy.ndarray): Centroids of clusters.

    Returns:
        numpy.ndarray: Cluster IDs for each data point.
    """
    labels = []
    for x in X:
        distances = [ComputeDistance(x, c) for c in centroids]
        labels.append(np.argmin(distances))
    return np.array(labels)


def compute_silhouette_coefficient(X, labels, k):
    """Compute the Silhouette coefficient for a clustering.

    Args:
        X (numpy.ndarray): Data points.
        labels (numpy.ndarray): Cluster labels for each data point.
        k (int): Number of clusters.

    Returns:
        float: Silhouette coefficient.
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

        # Check for zero division
        if a_i == 0 or b_i == 0:
            s_i = 0
        else:
            s_i = (b_i - a_i) / max(a_i, b_i)

        silhouette_coefficients.append(s_i)

    # If all data points belong to empty clusters, return 0
    if not silhouette_coefficients:
        return 0.0

    return np.nanmean(silhouette_coefficients)


def computeClusterRepresentatives(X, labels, k):
    """Compute cluster representatives (centroids).

    Args:
        X (numpy.ndarray): Data points.
        labels (numpy.ndarray): Cluster labels for each data point.
        k (int): Number of clusters.

    Returns:
        numpy.ndarray: Centroids of clusters.
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


def kmeans(X, k, maxIter=100):
    """
    Perform k-means clustering. This algorithm basically does iteratively determines the best
    K center point. This algorithm picks centroid locations to minimise the cumulative square of
    the distances from each datapoint to its closest centroid

    Args:
        X (numpy.ndarray): Data points.
        k (int): Number of clusters.
        maxIter (int, optional): Maximum number of iterations. Defaults to 100.

    Returns:
        numpy.ndarray: Cluster labels for each data point.
    """
    centroids = initialSelection(X, k)

    for _ in range(maxIter):
        labels = assignClusterIds(X, centroids)
        new_centroids = computeClusterRepresentatives(X, labels, k)

        if np.allclose(centroids, new_centroids):
            break

        centroids = new_centroids

    return labels


def plot_silhouette(silhouette_coefficients, ks):
    """Plot Silhouette coefficients for different values of k.

    Args:
        silhouette_coefficients (list): List of Silhouette coefficients.
        ks (range): Range of values for k.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(ks, silhouette_coefficients, marker='o')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Coefficient')
    plt.title('Silhouette Coefficient vs clusters - Question 1')
    plt.grid(True)
    plt.xticks(ks)
    plt.savefig('kmeans_silhouette_plot_Question1.png', format='png')  # Save the plot to the current folder
    plt.show()
    plt.close()


def load_dataset(file_path):
    """Load and preprocess the dataset.

    Args:
        file_path (str): Path to the dataset file.

    Returns:
        numpy.ndarray: Processed dataset.
    """
    try:
        data = pd.read_csv(file_path, header=None, sep='\s+')
    except FileNotFoundError:
        print("File not found.")
        return None
    except pd.errors.EmptyDataError:
        print("File is empty.")
        return None
    except pd.errors.ParserError:
        print("Error parsing the file.")
        return None

    if data.shape[0] < 2:
        print("Insufficient data points in the file.")
        return None

    X = data.iloc[:, 1:].values
    return X

"""Main Program"""

# Set a fixed seed value for reproducibility
np.random.seed(42)

# Filter out the runtime warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

file_path = 'dataset'
X = load_dataset(file_path)

if X is not None:

    # Compute Silhouette coefficients for different values of k
    silhouette_coefficients = []
    ks = range(1, 10)

    for k in ks:
        labels = kmeans(X, k)
        silhouette_coefficients.append(compute_silhouette_coefficient(X, labels, k))

    # Plot the results
    plot_silhouette(silhouette_coefficients, ks)

"""The graph depicts the correlation between the Silhouette Coefficient, a metric for assessing clustering quality, and the varying number of clusters (k) utilized in clustering analysis. The Silhouette Coefficient scale ranges from -1 to 1, with higher values denoting superior clustering quality.

Initially, there is a sharp decline in the Silhouette Coefficient as the number of clusters rises from 2 to 4, indicating that augmenting clusters from a single one does not notably enhance clustering quality. However, as the number of clusters escalates to approximately 5 or 6, the Silhouette Coefficient hits a local minimum, suggesting subpar clustering quality for that cluster count.

Interestingly, beyond 6 clusters, there's a turnaround: the Silhouette Coefficient starts to ascend, peaking at around 8 and 9 clusters however not higher than cluster 2.

Further analysis will help recommend the best cluster
"""