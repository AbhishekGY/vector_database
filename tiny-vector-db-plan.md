# 🗄️ Tiny Vector Database — Implementation Plan

## Overview

Build a vector database from scratch in pure Python that supports:
inserting vectors, quantizing them with Product Quantization, persisting the index to disk,
and querying for approximate nearest neighbours via a simple API.

No FAISS. No Pinecone. Every line written by hand.

---

## Project Structure

```
tiny-vector-db/
├── README.md
├── requirements.txt
│
├── core/
│   ├── __init__.py
│   ├── kmeans.py          # k-means clustering (builds codebooks)
│   ├── codebook.py        # Codebook class (train, encode, decode)
│   ├── pq.py              # Product Quantization (splits + encodes full vectors)
│   ├── index.py           # The main VectorIndex (insert, search, persist)
│   └── distance.py        # Distance functions (L2, dot product, cosine)
│
├── storage/
│   ├── __init__.py
│   └── disk.py            # Save/load index to disk (JSON + numpy .npy)
│
├── api/
│   ├── __init__.py
│   └── db.py              # Public-facing VectorDB class (the clean API)
│
├── tests/
│   ├── test_kmeans.py
│   ├── test_codebook.py
│   ├── test_pq.py
│   ├── test_index.py
│   └── test_db.py
│
└── notebooks/
    ├── 01_kmeans_demo.ipynb       # Visualise k-means convergence
    ├── 02_pq_demo.ipynb           # Step through PQ encoding by hand
    ├── 03_full_pipeline.ipynb     # End-to-end DB demo
    └── 04_benchmarks.ipynb        # Speed and recall benchmarks
```

---

## Phase 1 — Core Math (Days 1–2)

The foundation. Implement the raw algorithms with no abstraction — just functions operating on NumPy arrays.

### 1.1 Distance Functions (`core/distance.py`)

Start here — everything else depends on measuring distances.

```python
# What to implement:
def l2_squared(a, b)     # Squared Euclidean distance between two vectors
def l2_batch(query, matrix)   # Distance from one query to N vectors (vectorised)
def cosine_similarity(a, b)   # Optional stretch goal
```

**Test:** Verify `l2_squared([0,0], [3,4]) == 25.0`

---

### 1.2 K-Means (`core/kmeans.py`)

This is how codebooks are trained. Implement from scratch.

```python
# What to implement:
def initialise_centroids(vectors, k)     # Pick k random vectors as starting centroids
def assign_clusters(vectors, centroids)  # For each vector, find nearest centroid index
def update_centroids(vectors, assignments, k)  # Recompute centroid as mean of cluster
def kmeans(vectors, k, max_iters=100)    # Full training loop, returns final centroids
```

**Key detail:** Handle empty clusters (a centroid with no assigned vectors) — reinitialise it randomly.

**Test:** Run on 2D points, plot the clusters and centroids to verify visually.

---

### 1.3 Codebook (`core/codebook.py`)

Wraps k-means into a trained lookup table.

```python
class Codebook:
    def __init__(self, k=256)
    def train(self, subvectors)           # Run k-means, store centroids
    def encode(self, subvector)           # Return index of nearest centroid
    def encode_batch(self, subvectors)    # Vectorised version for N subvectors
    def decode(self, index)               # Return centroid vector for a given index
    def quantization_error(self, subvectors)  # Mean squared error after encoding
```

**Test:** Train on 1000 random 8-dim vectors with k=16. Encode one, decode it, measure error.

---

## Phase 2 — Product Quantization (Day 3)

Combine M codebooks to compress full vectors.

### 2.1 PQ Encoder (`core/pq.py`)

```python
class ProductQuantizer:
    def __init__(self, d, M, k=256)
    # d = vector dimension, M = number of subspaces, k = codebook size
    # subvector dimension = d // M

    def train(self, vectors)
    # Split vectors into M subspaces
    # Train one Codebook per subspace
    # Store all M codebooks

    def encode(self, vector)
    # Split vector into M subvectors
    # Encode each subvector with its codebook
    # Return array of M integer indices

    def encode_batch(self, vectors)
    # Encode N vectors, return (N x M) integer array

    def decode(self, codes)
    # Look up each index in its codebook
    # Concatenate centroids to reconstruct approximate vector

    def compute_distance_table(self, query)
    # Split query into M subvectors
    # For each subspace m, compute distance from query_m to all k centroids
    # Return (M x k) table — this is the key to fast search

    def approximate_distance(self, codes, distance_table)
    # For a stored code [i1, i2, ..., iM], sum distance_table[m][im] for all m
    # This is a table lookup, not a float computation
```

**Test:** Encode a known vector, decode it, measure reconstruction error. Verify it matches the hand-worked example from the conversation.

---

## Phase 3 — The Index (Day 4)

Store vectors and search over them.

### 3.1 Vector Index (`core/index.py`)

```python
class VectorIndex:
    def __init__(self, pq: ProductQuantizer)

    def add(self, vector_id, vector)
    # Encode vector → M codes
    # Append to codes table
    # Store id → row mapping

    def add_batch(self, ids, vectors)
    # Encode all vectors at once (faster)

    def search(self, query, top_k=10)
    # Compute distance table for query (once)
    # For every stored code, compute approximate distance via table lookup
    # Return top_k (id, distance) pairs sorted by distance

    def __len__(self)
    # Return number of indexed vectors
```

**Key insight:** `search` loops over N stored codes, but each distance computation is just M integer lookups + M additions. For M=8, that's 8 additions per vector — extremely fast.

**Test:** Insert 1000 random vectors. Query with one of them. Verify it returns itself as the top result.

---

## Phase 4 — Persistence (Day 5)

Save the index to disk and reload it.

### 4.1 Disk Storage (`storage/disk.py`)

```python
def save(index, path)
# Save codebook centroids as .npy files (efficient binary)
# Save PQ codes table as .npy
# Save id → row mapping as JSON
# Save config (d, M, k) as JSON

def load(path)
# Reconstruct ProductQuantizer from saved centroids
# Reconstruct VectorIndex from saved codes + mapping
# Return a ready-to-query VectorIndex
```

**Format on disk:**
```
my_index/
├── config.json          # d, M, k, num_vectors
├── codebooks.npy        # Shape: (M, k, d//M) — all centroids
├── codes.npy            # Shape: (N, M) — all encoded vectors
└── ids.json             # {row_index: vector_id}
```

**Test:** Save an index, delete it from memory, reload from disk, run the same query, verify identical results.

---

## Phase 5 — Public API (Day 6)

Wrap everything in a clean, simple interface.

### 5.1 VectorDB (`api/db.py`)

```python
class VectorDB:
    def __init__(self, dim, M=8, k=256)

    def fit(self, vectors)
    # Train PQ codebooks on a sample of vectors
    # Must be called before insert

    def insert(self, id, vector)
    # Add a single vector

    def insert_batch(self, ids, vectors)
    # Add many vectors efficiently

    def query(self, vector, top_k=5)
    # Return top_k most similar (id, score) pairs

    def save(self, path)
    def load(cls, path)   # classmethod
```

**Goal:** The entire database usable in 5 lines:

```python
db = VectorDB(dim=128, M=8, k=256)
db.fit(training_vectors)
db.insert_batch(ids, vectors)
results = db.query(my_vector, top_k=5)
db.save("./my_index")
```

---

## Phase 6 — Benchmarks & Notebooks (Day 7)

Validate that the system actually works well.

### 6.1 Recall@K

The key metric: of the true top-K nearest neighbours (computed by brute force), how many does the PQ index return?

```
Recall@10 = |PQ results ∩ True results| / 10
```

A good PQ index should achieve **Recall@10 > 0.90** on typical data.

### 6.2 Benchmark dimensions to measure

| Metric | How to measure |
|---|---|
| Recall@10 | Compare PQ results vs brute-force exact results |
| Index build time | Time `fit()` + `insert_batch()` |
| Query latency | Time `query()` for 1000 queries, report p50/p99 |
| Memory usage | `sys.getsizeof` on codes table vs original float array |
| Compression ratio | Original bytes / compressed bytes |

### 6.3 Ablation: vary M and k

Run a sweep and plot the tradeoff:

```
M = [4, 8, 16, 32]   x   k = [64, 128, 256]
→ plot Recall@10 vs Memory Usage
```

This gives you an empirical feel for the M and k tradeoffs we discussed in theory.

---

## Milestones & Checklist

```
Phase 1 — Core Math
  [ ] distance.py — l2_squared, l2_batch
  [ ] kmeans.py — full training loop with empty cluster handling
  [ ] codebook.py — train, encode, decode, quantization_error

Phase 2 — Product Quantization
  [ ] pq.py — train, encode_batch, decode, compute_distance_table
  [ ] Verify hand-worked example from notes matches code output

Phase 3 — Index
  [ ] index.py — add, add_batch, search
  [ ] Self-retrieval test (query returns itself as top result)

Phase 4 — Persistence
  [ ] disk.py — save, load
  [ ] Round-trip test (save → reload → same results)

Phase 5 — Public API
  [ ] db.py — clean VectorDB wrapper
  [ ] 5-line usage demo works end to end

Phase 6 — Benchmarks
  [ ] Recall@10 > 0.90 on test dataset
  [ ] Latency < 10ms for 10K vectors
  [ ] Compression ratio measured and documented
  [ ] M vs k tradeoff plot
```

---

## Stack

| Tool | Purpose |
|---|---|
| `numpy` | All vector math and array operations |
| `json` | Saving config and id mappings |
| `pathlib` | File path handling |
| `pytest` | Unit tests |
| `matplotlib` | Visualisations in notebooks |
| `jupyter` | Interactive notebooks |

No FAISS. No sklearn. No shortcuts. Pure NumPy + Python.

---

## Stretch Goals (after core is done)

- **Filtered search** — only search vectors matching a metadata filter (e.g. `category == "rock"`)
- **HNSW index** — replace brute-force scan with a graph-based index for sub-linear search
- **REST API** — wrap VectorDB in a FastAPI server with `/insert` and `/query` endpoints
- **Batch updates** — support deleting and updating vectors without full re-indexing
- **Benchmarks vs FAISS** — compare your implementation's recall and speed against the real thing

---

## References

- Jégou et al. (2011) — *Product Quantization for Nearest Neighbor Search* (the original PQ paper)
- [FAISS GitHub](https://github.com/facebookresearch/faiss) — read the source for inspiration
- Our conversation notes — *Vector Quantization Study Notes* (saved in Notion)
