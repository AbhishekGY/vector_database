import numpy as np
from core.pq import ProductQuantizer
from core.index import VectorIndex


def _make_trained_index(rng, n=200, d=32, M=4, k=16):
    """Helper: create a trained PQ and VectorIndex with n vectors."""
    data = rng.normal(size=(n, d))
    pq = ProductQuantizer(d=d, M=M, k=k)
    pq.train(data)
    index = VectorIndex(pq)
    return index, pq, data


def test_index_add_and_len():
    rng = np.random.default_rng(42)
    index, pq, data = _make_trained_index(rng)
    for i in range(5):
        index.add(f"vec_{i}", data[i])
    assert len(index) == 5


def test_index_add_batch():
    rng = np.random.default_rng(42)
    index, pq, data = _make_trained_index(rng, n=200)
    ids = [f"vec_{i}" for i in range(100)]
    index.add_batch(ids, data[:100])
    assert len(index) == 100
    assert index.codes.shape == (100, 4)


def test_index_add_then_add_batch():
    rng = np.random.default_rng(42)
    index, pq, data = _make_trained_index(rng)
    index.add("single", data[0])
    ids = [f"vec_{i}" for i in range(50)]
    index.add_batch(ids, data[1:51])
    assert len(index) == 51


def test_index_search_returns_correct_format():
    rng = np.random.default_rng(42)
    index, pq, data = _make_trained_index(rng)
    ids = [f"vec_{i}" for i in range(100)]
    index.add_batch(ids, data[:100])

    results = index.search(data[0], top_k=5)
    assert len(results) == 5
    for vid, dist in results:
        assert isinstance(vid, str)
        assert isinstance(dist, float)


def test_index_self_retrieval():
    rng = np.random.default_rng(42)
    data = rng.normal(size=(1000, 32))
    pq = ProductQuantizer(d=32, M=4, k=16)
    pq.train(data)

    index = VectorIndex(pq)
    ids = list(range(1000))
    index.add_batch(ids, data)

    # Query with vector 42 — it should return itself as top result
    results = index.search(data[42], top_k=10)
    assert results[0][0] == 42


def test_index_search_top_k_ordering():
    rng = np.random.default_rng(42)
    index, pq, data = _make_trained_index(rng)
    ids = list(range(200))
    index.add_batch(ids, data)

    results = index.search(data[0], top_k=10)
    distances = [d for _, d in results]
    assert distances == sorted(distances)


def test_index_search_top_k_exceeds_size():
    rng = np.random.default_rng(42)
    index, pq, data = _make_trained_index(rng)
    ids = list(range(5))
    index.add_batch(ids, data[:5])

    results = index.search(data[0], top_k=10)
    assert len(results) == 5
    distances = [d for _, d in results]
    assert distances == sorted(distances)
