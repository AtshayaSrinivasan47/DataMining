"""Atshaya Srinivasan - 201774445"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import warnings

"""
Question -4 Implement the Bisecting k-Means algorithm to compute a hierarchy of clustering that refines the initial (20)
single cluster to 9 clusters. For each s from 1 to 9, extract from the hierarchy of clustering the clustering
with s clusters and compute the Silhouette coefficient for this clustering. Plot s in the horizontal axis
and the Silhouette coefficient in the vertical axis in the same plot.
"""

def convert_to_2d_array(points):
    """
    Converts `points` to a 2-D numpy array.
    """
    points = np.array(points)
    if len(points.shape) == 1:
        points = np.expand_dims(points, -1)
    return points

def visualize_clusters(clusters):
    """
    Visualizes the first 2 dimensions of the data as a 2-D scatter plot.
    """
    plt.figure()
    for cluster in clusters:
        points = convert_to_2d_array(cluster)
        if points.shape[1] < 2:
            points = np.hstack([points, np.zeros_like(points)])
        plt.plot(points[:,0], points[:,1], 'o')
    plt.show()

def SSE(points):
    """
    Calculates the sum of squared errors for the given list of data points.
    """
    points = convert_to_2d_array(points)
    centroid = np.mean(points, 0)
    errors = np.linalg.norm(points-centroid, ord=2, axis=1)
    return np.sum(errors)

def kmeans(points, k=2, epochs=10, max_iter=100, verbose=False):
    """
    Clusters the list of points into `k` clusters using k-means clustering
    algorithm.
    """
    points = convert_to_2d_array(points)
    assert len(points) >= k, "Number of data points can't be less than k"

    best_sse = np.inf
    for ep in range(epochs):
        # Randomly initialize k centroids
        np.random.shuffle(points)
        centroids = points[0:k, :]

        last_sse = np.inf
        for it in range(max_iter):
            # Cluster assignment
            clusters = [None] * k
            for p in points:
                index = np.argmin(np.linalg.norm(centroids-p, 2, 1))
                if clusters[index] is None:
                    clusters[index] = np.expand_dims(p, 0)
                else:
                    clusters[index] = np.vstack((clusters[index], p))

            # Centroid update
            centroids = [np.mean(c, 0) for c in clusters]

            # SSE calculation
            sse = np.sum([SSE(c) for c in clusters])
            gain = last_sse - sse
            if verbose:
                print((f'Epoch: {ep:3d}, Iter: {it:4d}, '
                        f'SSE: {sse:12.4f}, Gain: {gain:12.4f}'))

            # Check for improvement
            if sse < best_sse:
                best_clusters, best_sse = clusters, sse

            # Epoch termination condition
            if np.isclose(gain, 0, atol=0.00001):
                break
            last_sse = sse

    return best_clusters

def bisecting_kmeans(points, k=2, epochs=10, max_iter=100, verbose=False):
    """
    Clusters the list of points into `k` clusters using bisecting k-means
    clustering algorithm. Internally, it uses the standard k-means with k=2 in
    each iteration.
    """
    points = convert_to_2d_array(points)
    clusters = [points]
    while len(clusters) < k:
        max_sse_i = np.argmax([SSE(c) for c in clusters])
        cluster = clusters.pop(max_sse_i)
        two_clusters = kmeans(cluster, k=2, epochs=epochs, max_iter=max_iter, verbose=verbose)
        clusters.extend(two_clusters)
    return clusters

# Function to compute the distance between two points
def computeDistance(x1, x2):
    """Calculate Euclidean distance between two points."""
    return np.sqrt(np.sum((x1 - x2) ** 2))

# Function to compute the Silhouette coefficient
def computeSilhouetteCoefficient(X, labels, k):
    """Compute the Silhouette coefficient.

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
        a_i = np.mean([computeDistance(X[i], c) for c in cluster_i if not np.array_equal(X[i], c)])

        # Compute b_i (mean nearest-cluster distance)
        b_i = np.inf
        for j in range(k):
            if j != labels[i]:
                cluster_j = X[labels == j]
                if cluster_j.size > 0:
                    b_i = min(b_i, np.mean([computeDistance(X[i], c) for c in cluster_j]))

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

    return np.mean(silhouette_coefficients)

# Function to plot Silhouette coefficients
def plotSilhouette(silhouette_coefficients, ks):
    """Plot Silhouette coefficients for different numbers of clusters.

    Args:
        silhouette_coefficients (list): List of Silhouette coefficients.
        ks (list): List of values for the number of clusters.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(ks, silhouette_coefficients, marker='o')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Coefficient')
    plt.title('Silhouette Coefficient vs clusters - Question 4')
    plt.grid(True)
    plt.xticks(ks)
    plt.savefig('Question_4_silhoutte graph.png')
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


# Set the random seed for reproducibility
np.random.seed(42)


# Filter out the runtime warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

file_path = 'dataset'
X = load_dataset(file_path)

if X is not None:

    # Perform Bisecting k-Means algorithm
    clusterings = bisecting_kmeans(X, k=9, epochs=10, max_iter=100, verbose=False)

    # Compute Silhouette coefficient for each clustering
    silhouette_coefficients = []
    for s in range(1, 10):
        labels = [int(c in clusterings[s - 1]) for c in X]
        silhouette_coefficients.append(computeSilhouetteCoefficient(X, np.array(labels), s))

    # Plot the results
    plotSilhouette(silhouette_coefficients, list(range(1, 10)))

"""The graph presents an intriguing trend. Initially, with a small number of clusters (k=2), the Silhouette Coefficient is relatively high, approximately 0.06. However, as the cluster count rises from 2 to 3, the Silhouette Coefficient steadily declines, bottoming out around 0.02 at k=3.

Yet, as the clusters expand further from 3 to 5, the Silhouette Coefficient begins to rise again, peaking at roughly 0.03 for k=5. This indicates that for this dataset, the optimal cluster count might be around 5.

However, beyond k=5, the Silhouette Coefficient appears to decrease once more, suggesting that increasing clusters may not enhance clustering quality.

The analysis implies that the ideal cluster count for this dataset probably falls between 2 and 5, with a potential sweet spot around 5 clusters based on the local peak in the Silhouette Coefficient. However, further examination and domain expertise would be necessary for a definitive conclusion.
"""