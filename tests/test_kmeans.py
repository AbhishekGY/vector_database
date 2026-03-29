import numpy as np
from core.distance import l2_squared, l2_batch, cosine_similarity
from core.kmeans import initialise_centroids, assign_clusters, update_centroids, kmeans


# --- Distance tests ---

def test_l2_squared_known_value():
    assert l2_squared(np.array([0.0, 0.0]), np.array([3.0, 4.0])) == 25.0


def test_l2_squared_identical():
    a = np.array([1.0, 2.0, 3.0])
    assert l2_squared(a, a) == 0.0


def test_l2_batch_matches_manual():
    query = np.array([0.0, 0.0])
    matrix = np.array([[3.0, 4.0], [1.0, 0.0], [0.0, 1.0]])
    result = l2_batch(query, matrix)
    np.testing.assert_array_almost_equal(result, [25.0, 1.0, 1.0])


def test_l2_batch_shape():
    result = l2_batch(np.zeros(8), np.random.randn(100, 8))
    assert result.shape == (100,)


def test_cosine_similarity_parallel():
    a = np.array([1.0, 0.0])
    assert cosine_similarity(a, a) == 1.0


def test_cosine_similarity_orthogonal():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert abs(cosine_similarity(a, b)) < 1e-10


def test_cosine_similarity_zero_vector():
    assert cosine_similarity(np.zeros(3), np.array([1.0, 2.0, 3.0])) == 0.0


# --- K-means tests ---

def test_initialise_centroids_shape():
    vectors = np.random.randn(50, 4)
    centroids = initialise_centroids(vectors, 5)
    assert centroids.shape == (5, 4)


def test_initialise_centroids_from_data():
    vectors = np.random.randn(50, 4)
    centroids = initialise_centroids(vectors, 5)
    # Every centroid should be an actual row from vectors
    for c in centroids:
        assert any(np.allclose(c, v) for v in vectors)


def test_assign_clusters_simple():
    vectors = np.array([[0.0, 0.0], [10.0, 0.0], [0.1, 0.1]])
    centroids = np.array([[0.0, 0.0], [10.0, 0.0]])
    assignments = assign_clusters(vectors, centroids)
    np.testing.assert_array_equal(assignments, [0, 1, 0])


def test_kmeans_returns_correct_shape():
    vectors = np.random.randn(200, 5)
    centroids = kmeans(vectors, k=8, max_iters=10)
    assert centroids.shape == (8, 5)


def test_kmeans_converges_on_separated_clusters():
    rng = np.random.default_rng(42)
    c1 = rng.normal(loc=[0, 0], scale=0.1, size=(100, 2))
    c2 = rng.normal(loc=[10, 0], scale=0.1, size=(100, 2))
    c3 = rng.normal(loc=[0, 10], scale=0.1, size=(100, 2))
    vectors = np.vstack([c1, c2, c3])

    # k-means is stochastic; retry up to 5 times (standard for random init)
    true_centers = np.array([[0, 0], [10, 0], [0, 10]], dtype=float)
    for attempt in range(5):
        centroids = kmeans(vectors, k=3)
        assert centroids.shape == (3, 2)
        all_close = all(
            min(l2_squared(tc, c) for c in centroids) < 1.0
            for tc in true_centers
        )
        if all_close:
            break
    assert all_close, "k-means did not converge to true centers after 5 attempts"
