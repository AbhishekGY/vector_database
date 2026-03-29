import numpy as np
import pytest
from api.db import VectorDB


def test_vectordb_five_line_demo(tmp_path):
    """The 5-line usage from the plan works end to end."""
    rng = np.random.default_rng(42)
    training_vectors = rng.normal(size=(500, 128))
    ids = list(range(500))

    db = VectorDB(dim=128, M=8, k=16)
    db.fit(training_vectors)
    db.insert_batch(ids, training_vectors)
    results = db.query(training_vectors[0], top_k=5)
    db.save(str(tmp_path / "my_index"))

    assert len(results) == 5
    assert results[0][0] == 0  # self-retrieval


def test_vectordb_insert_before_fit():
    rng = np.random.default_rng(42)
    db = VectorDB(dim=32, M=4, k=16)
    with pytest.raises(RuntimeError):
        db.insert("a", rng.normal(size=32))


def test_vectordb_insert_batch_before_fit():
    rng = np.random.default_rng(42)
    db = VectorDB(dim=32, M=4, k=16)
    with pytest.raises(RuntimeError):
        db.insert_batch(["a"], rng.normal(size=(1, 32)))


def test_vectordb_single_insert_and_query():
    rng = np.random.default_rng(42)
    data = rng.normal(size=(200, 32))
    db = VectorDB(dim=32, M=4, k=16)
    db.fit(data)
    for i in range(50):
        db.insert(f"v{i}", data[i])
    results = db.query(data[0], top_k=3)
    assert len(results) == 3
    assert results[0][0] == "v0"


def test_vectordb_load_classmethod(tmp_path):
    rng = np.random.default_rng(42)
    data = rng.normal(size=(200, 32))

    db = VectorDB(dim=32, M=4, k=16)
    db.fit(data)
    db.insert_batch(list(range(200)), data)
    results_before = db.query(data[5], top_k=5)
    db.save(str(tmp_path / "idx"))

    loaded = VectorDB.load(str(tmp_path / "idx"))
    assert loaded.dim == 32
    assert loaded.M == 4
    assert loaded.k == 16
    assert loaded._trained is True

    results_after = loaded.query(data[5], top_k=5)
    assert results_before == results_after


def test_vectordb_query_ordering():
    rng = np.random.default_rng(42)
    data = rng.normal(size=(300, 32))
    db = VectorDB(dim=32, M=4, k=16)
    db.fit(data)
    db.insert_batch(list(range(300)), data)
    results = db.query(data[0], top_k=10)
    distances = [d for _, d in results]
    assert distances == sorted(distances)
