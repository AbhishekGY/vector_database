import numpy as np
from core.distance import l2_batch


def initialise_centroids(vectors, k):
    """Pick k random vectors as starting centroids."""
    n = len(vectors)
    if k > n:
        raise ValueError(f"k ({k}) must be <= number of vectors ({n})")
    indices = np.random.choice(n, size=k, replace=False)
    return vectors[indices].copy()


def assign_clusters(vectors, centroids):
    """For each vector, find the index of the nearest centroid.

    Loops over k centroids (not n vectors) for memory efficiency.

    Args:
        vectors: shape (n, d)
        centroids: shape (k, d)

    Returns:
        shape (n,) integer array of cluster assignments
    """
    distances = np.stack([l2_batch(c, vectors) for c in centroids])  # (k, n)
    return np.argmin(distances, axis=0)


def update_centroids(vectors, assignments, k):
    """Recompute each centroid as the mean of its assigned vectors.

    Empty clusters are reinitialised with a random data point.
    """
    n, d = vectors.shape
    centroids = np.empty((k, d))
    for i in range(k):
        mask = assignments == i
        if mask.sum() == 0:
            centroids[i] = vectors[np.random.randint(n)]
        else:
            centroids[i] = vectors[mask].mean(axis=0)
    return centroids


def kmeans(vectors, k, max_iters=100):
    """Run k-means clustering. Returns final centroids of shape (k, d)."""
    centroids = initialise_centroids(vectors, k)
    for _ in range(max_iters):
        assignments = assign_clusters(vectors, centroids)
        new_centroids = update_centroids(vectors, assignments, k)
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    return centroids
