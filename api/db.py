from core.pq import ProductQuantizer
from core.index import VectorIndex
from storage.disk import save as disk_save, load as disk_load


class VectorDB:
    """Clean public API for the vector database.

    Usage:
        db = VectorDB(dim=128, M=8, k=256)
        db.fit(training_vectors)
        db.insert_batch(ids, vectors)
        results = db.query(my_vector, top_k=5)
        db.save("./my_index")
    """

    def __init__(self, dim, M=8, k=256):
        self.dim = dim
        self.M = M
        self.k = k
        self._pq = ProductQuantizer(dim, M, k)
        self._index = VectorIndex(self._pq)
        self._trained = False

    def fit(self, vectors):
        """Train PQ codebooks on a sample of vectors. Must be called before insert."""
        self._pq.train(vectors)
        self._trained = True

    def insert(self, id, vector):
        """Add a single vector."""
        if not self._trained:
            raise RuntimeError("Must call fit() before insert()")
        self._index.add(id, vector)

    def insert_batch(self, ids, vectors):
        """Add many vectors efficiently."""
        if not self._trained:
            raise RuntimeError("Must call fit() before insert_batch()")
        self._index.add_batch(ids, vectors)

    def query(self, vector, top_k=5):
        """Return top_k most similar (id, distance) pairs, sorted by distance."""
        return self._index.search(vector, top_k=top_k)

    def save(self, path):
        """Persist the index to disk."""
        disk_save(self._index, path)

    @classmethod
    def load(cls, path):
        """Load a saved index from disk. Returns a ready-to-query VectorDB."""
        index = disk_load(path)
        pq = index.pq
        db = cls.__new__(cls)
        db.dim = pq.d
        db.M = pq.M
        db.k = pq.k
        db._pq = pq
        db._index = index
        db._trained = True
        return db
