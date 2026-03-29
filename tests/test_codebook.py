import numpy as np
from core.codebook import Codebook


def test_codebook_train_sets_centroids():
    rng = np.random.default_rng(42)
    cb = Codebook(k=16)
    cb.train(rng.normal(size=(1000, 8)))
    assert cb.centroids.shape == (16, 8)


def test_codebook_encode_returns_valid_index():
    rng = np.random.default_rng(42)
    data = rng.normal(size=(1000, 8))
    cb = Codebook(k=16)
    cb.train(data)
    code = cb.encode(data[0])
    assert isinstance(code, int)
    assert 0 <= code < 16


def test_codebook_decode_returns_correct_shape():
    rng = np.random.default_rng(42)
    cb = Codebook(k=16)
    cb.train(rng.normal(size=(1000, 8)))
    decoded = cb.decode(0)
    assert decoded.shape == (8,)


def test_codebook_encode_batch_matches_single():
    rng = np.random.default_rng(42)
    data = rng.normal(size=(100, 8))
    cb = Codebook(k=16)
    cb.train(data)

    batch_codes = cb.encode_batch(data[:10])
    single_codes = np.array([cb.encode(sv) for sv in data[:10]])
    np.testing.assert_array_equal(batch_codes, single_codes)


def test_codebook_quantization_error_bounded():
    rng = np.random.default_rng(42)
    data = rng.normal(size=(1000, 8))
    cb = Codebook(k=16)
    cb.train(data)

    error = cb.quantization_error(data)
    assert error > 0
    # Error should be less than the variance of the data (~8.0 for 8-dim standard normal)
    assert error < 8.0


def test_codebook_decode_does_not_mutate_centroids():
    rng = np.random.default_rng(42)
    cb = Codebook(k=16)
    cb.train(rng.normal(size=(200, 4)))

    original = cb.centroids[0].copy()
    decoded = cb.decode(0)
    decoded[:] = 999.0  # mutate the returned array
    np.testing.assert_array_equal(cb.centroids[0], original)
