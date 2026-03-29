import numpy as np


class VectorIndex:
    """Stores PQ-encoded vectors and performs approximate nearest-neighbor search."""

    def __init__(self, pq):
        self.pq = pq
        self.codes = None  # shape (n, M) uint8
        self.ids = []

    def add(self, vector_id, vector):
        """Encode and add a single vector to the index."""
        code = self.pq.encode(vector).reshape(1, -1)
        if self.codes is None:
            self.codes = code
        else:
            self.codes = np.vstack([self.codes, code])
        self.ids.append(vector_id)

    def add_batch(self, ids, vectors):
        """Encode and add multiple vectors to the index."""
        new_codes = self.pq.encode_batch(vectors)
        if self.codes is None:
            self.codes = new_codes
        else:
            self.codes = np.vstack([self.codes, new_codes])
        self.ids.extend(ids)

    def search(self, query, top_k=10):
        """Search for the top_k most similar vectors to the query.

        Returns:
            list of (vector_id, distance) tuples sorted by ascending distance
        """
        table = self.pq.compute_distance_table(query)
        dists = self.pq.approximate_distance(self.codes, table)

        n = len(self.ids)
        if top_k >= n:
            top_indices = np.argsort(dists)
        else:
            top_indices = np.argpartition(dists, top_k)[:top_k]
            top_indices = top_indices[np.argsort(dists[top_indices])]

        return [(self.ids[i], float(dists[i])) for i in top_indices]

    def __len__(self):
        return len(self.ids)
