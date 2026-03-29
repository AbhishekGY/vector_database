import numpy as np
import pytest
from core.pq import ProductQuantizer
from core.distance import l2_squared


def test_pq_init_validates_dimensions():
    pq = ProductQuantizer(d=128, M=8)
    assert pq.d_sub == 16
    with pytest.raises(ValueError):
        ProductQuantizer(d=10, M=3)


def test_pq_train_creates_codebooks():
    rng = np.random.default_rng(42)
    pq = ProductQuantizer(d=32, M=4, k=16)
    pq.train(rng.normal(size=(500, 32)))
    assert len(pq.codebooks) == 4
    for cb in pq.codebooks:
        assert cb.centroids.shape == (16, 8)


def test_pq_encode_returns_correct_shape():
    rng = np.random.default_rng(42)
    data = rng.normal(size=(100, 32))
    pq = ProductQuantizer(d=32, M=4, k=16)
    pq.train(data)

    single = pq.encode(data[0])
    assert single.shape == (4,)
    assert single.dtype == np.uint8
    assert np.all(single < 16)

    batch = pq.encode_batch(data[:50])
    assert batch.shape == (50, 4)
    assert batch.dtype == np.uint8
    assert np.all(batch < 16)


def test_pq_decode_reconstructs_approximate_vector():
    rng = np.random.default_rng(42)
    data = rng.normal(size=(500, 32))
    pq = ProductQuantizer(d=32, M=4, k=16)
    pq.train(data)

    codes = pq.encode(data[0])
    reconstructed = pq.decode(codes)
    assert reconstructed.shape == (32,)

    error = l2_squared(data[0], reconstructed)
    # Error should be bounded — less than data variance (~32 for 32-dim standard normal)
    assert error < 32.0


def test_pq_encode_batch_matches_single():
    rng = np.random.default_rng(42)
    data = rng.normal(size=(200, 32))
    pq = ProductQuantizer(d=32, M=4, k=16)
    pq.train(data)

    batch_codes = pq.encode_batch(data[:10])
    single_codes = np.stack([pq.encode(v) for v in data[:10]])
    np.testing.assert_array_equal(batch_codes, single_codes)


def test_pq_distance_table_shape():
    rng = np.random.default_rng(42)
    data = rng.normal(size=(200, 32))
    pq = ProductQuantizer(d=32, M=4, k=16)
    pq.train(data)

    table = pq.compute_distance_table(data[0])
    assert table.shape == (4, 16)
    assert np.all(table >= 0)


def test_pq_approximate_distance_single():
    rng = np.random.default_rng(42)
    data = rng.normal(size=(200, 32))
    pq = ProductQuantizer(d=32, M=4, k=16)
    pq.train(data)

    codes = pq.encode(data[1])
    table = pq.compute_distance_table(data[0])
    dist = pq.approximate_distance(codes, table)
    assert isinstance(dist, float)
    assert dist >= 0


def test_pq_approximate_distance_batch():
    rng = np.random.default_rng(42)
    data = rng.normal(size=(200, 32))
    pq = ProductQuantizer(d=32, M=4, k=16)
    pq.train(data)

    all_codes = pq.encode_batch(data)
    table = pq.compute_distance_table(data[0])
    dists = pq.approximate_distance(all_codes, table)
    assert dists.shape == (200,)
    assert np.all(dists >= 0)


def test_pq_approximate_distance_vs_exact():
    rng = np.random.default_rng(42)
    data = rng.normal(size=(100, 32))
    pq = ProductQuantizer(d=32, M=4, k=16)
    pq.train(data)

    query = data[0]
    all_codes = pq.encode_batch(data)
    table = pq.compute_distance_table(query)
    approx_dists = pq.approximate_distance(all_codes, table)

    # Exact distances
    exact_dists = np.array([l2_squared(query, v) for v in data])

    # The approximate top-1 should be among the exact top-5
    approx_top1 = np.argmin(approx_dists)
    exact_top5 = np.argsort(exact_dists)[:5]
    assert approx_top1 in exact_top5


def test_pq_roundtrip_error():
    rng = np.random.default_rng(42)
    data = rng.normal(size=(1000, 128))
    pq = ProductQuantizer(d=128, M=8, k=16)
    pq.train(data)

    codes = pq.encode_batch(data)
    reconstructed = pq.decode(codes)
    assert reconstructed.shape == (1000, 128)

    errors = np.sum((data - reconstructed) ** 2, axis=1)
    mean_error = np.mean(errors)
    assert mean_error > 0
    # Mean error should be less than data variance (~128 for 128-dim standard normal)
    assert mean_error < 128.0
