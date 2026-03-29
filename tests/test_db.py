import numpy as np
from core.pq import ProductQuantizer
from core.index import VectorIndex
from storage.disk import save, load


def _make_index_with_data(rng, n=100, d=32, M=4, k=16):
    """Helper: create a trained index with n vectors inserted."""
    data = rng.normal(size=(n, d))
    pq = ProductQuantizer(d=d, M=M, k=k)
    pq.train(data)
    index = VectorIndex(pq)
    ids = [f"vec_{i}" for i in range(n)]
    index.add_batch(ids, data)
    return index, data


def test_save_creates_files(tmp_path):
    rng = np.random.default_rng(42)
    index, _ = _make_index_with_data(rng)
    save(index, tmp_path / "idx")

    assert (tmp_path / "idx" / "config.json").exists()
    assert (tmp_path / "idx" / "codebooks.npy").exists()
    assert (tmp_path / "idx" / "codes.npy").exists()
    assert (tmp_path / "idx" / "ids.json").exists()


def test_save_load_roundtrip(tmp_path):
    rng = np.random.default_rng(42)
    index, data = _make_index_with_data(rng)

    # Query before save
    results_before = index.search(data[0], top_k=5)

    save(index, tmp_path / "idx")
    del index

    loaded = load(tmp_path / "idx")
    results_after = loaded.search(data[0], top_k=5)

    assert results_before == results_after


def test_load_config_matches(tmp_path):
    rng = np.random.default_rng(42)
    index, _ = _make_index_with_data(rng, d=32, M=4, k=16)
    save(index, tmp_path / "idx")

    loaded = load(tmp_path / "idx")
    assert loaded.pq.d == 32
    assert loaded.pq.M == 4
    assert loaded.pq.k == 16


def test_load_preserves_ids(tmp_path):
    rng = np.random.default_rng(42)
    index, _ = _make_index_with_data(rng, n=50)
    original_ids = list(index.ids)

    save(index, tmp_path / "idx")
    loaded = load(tmp_path / "idx")

    assert loaded.ids == original_ids


def test_load_preserves_codes_shape(tmp_path):
    rng = np.random.default_rng(42)
    index, _ = _make_index_with_data(rng, n=80, M=4)
    save(index, tmp_path / "idx")

    loaded = load(tmp_path / "idx")
    assert loaded.codes.shape == (80, 4)
    assert loaded.codes.dtype == np.uint8


def test_save_load_search_identical(tmp_path):
    rng = np.random.default_rng(42)
    index, data = _make_index_with_data(rng, n=200)
    save(index, tmp_path / "idx")
    loaded = load(tmp_path / "idx")

    # Run 10 different queries, all must match exactly
    for i in range(0, 100, 10):
        results_orig = index.search(data[i], top_k=5)
        results_loaded = loaded.search(data[i], top_k=5)
        assert results_orig == results_loaded
