import numpy as np
from core.codebook import Codebook
from core.distance import l2_batch


class ProductQuantizer:
    """Product Quantization: splits D-dim vectors into M subspaces,
    trains one codebook per subspace, and encodes vectors as M uint8 indices."""

    def __init__(self, d, M, k=256):
        if d % M != 0:
            raise ValueError(f"d ({d}) must be divisible by M ({M})")
        self.d = d
        self.M = M
        self.k = k
        self.d_sub = d // M
        self.codebooks = None

    def _split(self, vectors):
        """Split vectors into M subvector groups along the feature axis.

        Args:
            vectors: shape (n, d) or (d,) for a single vector

        Returns:
            list of M arrays, each shape (n, d_sub) or (d_sub,) for single
        """
        single = vectors.ndim == 1
        if single:
            vectors = vectors.reshape(1, -1)
        parts = np.split(vectors, self.M, axis=1)
        if single:
            parts = [p.squeeze(axis=0) for p in parts]
        return parts

    def train(self, vectors):
        """Train M codebooks, one per subspace."""
        parts = self._split(vectors)
        self.codebooks = []
        for m in range(self.M):
            cb = Codebook(k=self.k)
            cb.train(parts[m])
            self.codebooks.append(cb)

    def encode(self, vector):
        """Encode a single vector into M uint8 codes."""
        parts = self._split(vector)
        codes = [self.codebooks[m].encode(parts[m]) for m in range(self.M)]
        return np.array(codes, dtype=np.uint8)

    def encode_batch(self, vectors):
        """Encode N vectors into an (N, M) uint8 code array."""
        n = len(vectors)
        parts = self._split(vectors)
        codes = np.empty((n, self.M), dtype=np.uint8)
        for m in range(self.M):
            codes[:, m] = self.codebooks[m].encode_batch(parts[m])
        return codes

    def decode(self, codes):
        """Reconstruct approximate vector(s) from codes.

        Args:
            codes: shape (M,) for single or (n, M) for batch

        Returns:
            shape (d,) or (n, d) reconstructed vectors
        """
        if codes.ndim == 1:
            parts = [self.codebooks[m].decode(codes[m]) for m in range(self.M)]
            return np.concatenate(parts)
        else:
            parts = [self.codebooks[m].centroids[codes[:, m]] for m in range(self.M)]
            return np.concatenate(parts, axis=1)

    def compute_distance_table(self, query):
        """Precompute distance table for a query vector.

        Returns:
            shape (M, k) table where table[m][i] is the squared L2 distance
            from query's m-th subvector to centroid i in codebook m.
        """
        parts = self._split(query)
        table = np.empty((self.M, self.k))
        for m in range(self.M):
            table[m] = l2_batch(parts[m], self.codebooks[m].centroids)
        return table

    def approximate_distance(self, codes, distance_table):
        """Compute approximate distances using precomputed distance table.

        Args:
            codes: shape (M,) for single or (n, M) for batch
            distance_table: shape (M, k) from compute_distance_table

        Returns:
            scalar float for single, shape (n,) array for batch
        """
        if codes.ndim == 1:
            return float(sum(distance_table[m, codes[m]] for m in range(self.M)))
        else:
            m_idx = np.arange(self.M).reshape(-1, 1)
            return distance_table[m_idx, codes.T].sum(axis=0)
