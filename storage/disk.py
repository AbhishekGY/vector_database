import json
from pathlib import Path

import numpy as np

from core.pq import ProductQuantizer
from core.codebook import Codebook
from core.index import VectorIndex


def save(index, path):
    """Save a VectorIndex to disk.

    Creates a directory with config.json, codebooks.npy, codes.npy, and ids.json.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    pq = index.pq
    config = {"d": pq.d, "M": pq.M, "k": pq.k, "num_vectors": len(index)}
    with open(path / "config.json", "w") as f:
        json.dump(config, f)

    # Stack all codebook centroids: M arrays of (k, d_sub) → (M, k, d_sub)
    codebooks_array = np.stack([cb.centroids for cb in pq.codebooks])
    np.save(path / "codebooks.npy", codebooks_array)

    np.save(path / "codes.npy", index.codes)

    with open(path / "ids.json", "w") as f:
        json.dump(index.ids, f)


def load(path):
    """Load a VectorIndex from disk. Returns a ready-to-query VectorIndex."""
    path = Path(path)

    with open(path / "config.json") as f:
        config = json.load(f)
    d, M, k = config["d"], config["M"], config["k"]

    codebooks_array = np.load(path / "codebooks.npy")

    pq = ProductQuantizer(d, M, k)
    pq.codebooks = []
    for m in range(M):
        cb = Codebook(k=k)
        cb.centroids = codebooks_array[m]
        pq.codebooks.append(cb)

    index = VectorIndex(pq)
    index.codes = np.load(path / "codes.npy")
    with open(path / "ids.json") as f:
        index.ids = json.load(f)

    return index
