import numpy as np
from core.kmeans import kmeans, assign_clusters
from core.distance import l2_batch


class Codebook:
    """A codebook trained via k-means that can encode subvectors to centroid indices."""

    def __init__(self, k=256):
        self.k = k
        self.centroids = None  # shape (k, d_sub) after training

    def train(self, subvectors):
        """Train the codebook by running k-means on the given subvectors."""
        self.centroids = kmeans(subvectors, self.k)

    def encode(self, subvector):
        """Return the index of the nearest centroid."""
        distances = l2_batch(subvector, self.centroids)
        return int(np.argmin(distances))

    def encode_batch(self, subvectors):
        """Encode N subvectors at once. Returns shape (n,) integer array."""
        return assign_clusters(subvectors, self.centroids)

    def decode(self, index):
        """Return the centroid vector for a given index."""
        return self.centroids[index].copy()

    def quantization_error(self, subvectors):
        """Mean squared L2 error between original and reconstructed subvectors."""
        codes = self.encode_batch(subvectors)
        reconstructed = self.centroids[codes]
        return float(np.mean(np.sum((subvectors - reconstructed) ** 2, axis=1)))
