import numpy as np


def l2_squared(a, b):
    """Squared Euclidean distance between two vectors."""
    return float(np.sum((a - b) ** 2))


def l2_batch(query, matrix):
    """Squared Euclidean distance from one query to N vectors (vectorised).

    Args:
        query: shape (d,)
        matrix: shape (n, d)

    Returns:
        shape (n,) array of squared distances
    """
    return np.sum((matrix - query) ** 2, axis=1)


def cosine_similarity(a, b):
    """Cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))
