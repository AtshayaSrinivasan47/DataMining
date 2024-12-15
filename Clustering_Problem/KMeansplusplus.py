# Atshaya Srinivasan - 201774445

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import warnings

"""
Question 3 - Implement k-means++ clustering algorithm and cluster the dataset provided using it. Vary the value (20)
of k from 1 to 9 and compute the Silhouette coefficient for each set of clusters. Plot k in the horizontal
axis and the Silhouette coefficient in the vertical axis in the same plot.
"""

def ComputeDistance(x1, x2):
    """
    Input:
    x1: numpy array, coordinates of the first point
    x2: numpy array, coordinates of the second point

    Output:
    distance: float, Euclidean distance between the two points

    This function calculates the Euclidean distance between two points.
    """
    return np.sqrt(np.sum((x1 - x2) ** 2))


def initialSelection(X, k):
    """
    Input:
    X: numpy array, data points
    k: int, number of centroids

    Output:
    centroids: numpy array, initialized centroids

    This function initializes centroids using the k-means++ approach.
    This approach is introduced to overcome the disadvantage of sensitivity when randomly choosing centroid using K means approach.
    """
    centroids = []
    centroids.append(X[np.random.randint(X.shape[0])])

    for _ in range(1, k):
        distances = np.array([min([ComputeDistance(x, c) for c in centroids]) for x in X])
        probabilities = distances / distances.sum()
        centroids.append(X[np.random.choice(range(X.shape[0]), p=probabilities)])

    return np.array(centroids)


def assignClusterIds(X, centroids):
    """
    Input:
    X: numpy array, data points
    centroids: numpy array, centroids

    Output:
    labels: numpy array, cluster ids assigned to each data point

    This function assigns cluster ids to each data point based on the nearest centroid.
    """
    labels = []
    for x in X:
        distances = [ComputeDistance(x, c) for c in centroids]
        labels.append(np.argmin(distances))
    return np.array(labels)


def computeClusterRepresentatives(X, labels, k):
    """
    Input:
    X: numpy array, data points
    labels: numpy array, cluster ids assigned to each data point
    k: int, number of clusters

    Output:
    centroids: numpy array, cluster representatives

    This function computes the cluster representatives (centroids) based on the mean of data points in each cluster.
    """
    centroids = []
    for i in range(k):
        cluster_i = X[labels == i]
        if cluster_i.size == 0:
            continue
        centroid_i = np.mean(cluster_i, axis=0)
        centroids.append(centroid_i)
    return np.array(centroids)


def compute_silhouette_coefficient(X, labels, k):
    """
    Input:
    X: numpy array, data points
    labels: numpy array, cluster ids assigned to each data point
    k: int, number of clusters

    Output:
    silhouette_coefficient: float, Silhouette coefficient

    This function computes the Silhouette coefficient for a given clustering.
    """
    n = X.shape[0]
    silhouette_coefficients = []

    for i in range(n):
        cluster_i = X[labels == labels[i]]

        # Skip data points belonging to empty clusters
        if cluster_i.size == 0:
            continue

        # Compute a_i (mean intra-cluster distance)
        a_i = np.mean([ComputeDistance(X[i], c) for c in cluster_i if np.not_equal(X[i], c).any()])

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

        s_i = (b_i - a_i) / max(a_i, b_i)
        silhouette_coefficients.append(s_i)

    # If all data points belong to empty clusters, return 0
    if not silhouette_coefficients:
        return 0.0

    return np.mean(silhouette_coefficients)


def kmeans_plusplus(X, k, maxIter=1000):
    """
    Input:
    X: numpy array, data points
    k: int, number of clusters
    maxIter: int, maximum number of iterations

    Output:
    labels: numpy array, cluster ids assigned to each data point

    This function performs k-means++ clustering.
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
    """
    Input:
    silhouette_coefficients: list, Silhouette coefficients for different values of k
    ks: list, range of values of k

    Output:
    None

    This function plots the Silhouette coefficients for different values of k.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(ks, silhouette_coefficients, marker='o')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Coefficient')
    plt.title('Silhouette Coefficient vs clusters - Question 3')
    plt.grid(True)
    plt.xticks(ks)
    plt.savefig('Question3_Silhouette_Plot.png')
    plt.show()
    plt.close()

def load_dataset(filename):
    """
    Input:
    filename: str, name of the file containing the dataset

    Output:
    X: numpy array, data points

    This function loads the dataset from the specified file and processes it for further use.
    """
    try:
        data = pd.read_csv(filename, header=None, sep='\s+')
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

# Set a fixed seed for reproducibility
SEED = 42
np.random.seed(SEED)

# Filter out the runtime warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

filename = 'dataset'
X = load_dataset(filename)

if X is not None:
    # Compute Silhouette coefficients for different values of k
    silhouette_coefficients = []
    ks = range(1, 10)

    for k in ks:
        labels = kmeans_plusplus(X, k)
        silhouette_coefficients.append(compute_silhouette_coefficient(X, labels, k))

    # Plot the results
    plot_silhouette(silhouette_coefficients, ks)

"""
The plot displays the Silhouette coefficients for different values of k (number of clusters) ranging from 1 to 9.

# **Observations from the plot:**

*   The highest Silhouette coefficient is achieved when k = 2, with a value of approximately 0.15. This suggests that the optimal number of clusters for the given dataset is 2, as it results in the most cohesive and well-separated clustering.
*   As k increases beyond 2, the Silhouette coefficient generally decreases, indicating that the clustering quality deteriorates as the number of clusters becomes larger.
*   There are some fluctuations in the Silhouette coefficient values, with local peaks observed at k = 4 and k = 5, but these values are lower than the peak at k = 2.
*   For larger values of k (e.g., k = 9), the Silhouette coefficient drops to around 0.07, suggesting that a higher number of clusters may not be appropriate for this dataset.

Based on the analysis of the Silhouette plot, the recommended number of clusters for the given dataset is 2."""