# ⚡ Fast Proximity v3.0

> **High-performance, JIT-accelerated N-Dimensional KNN search engine.**  
> Returns *exactly* the K nearest neighbors under full Euclidean distance, within radius R — minimizing average query time for datasets of size N in D dimensions.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [How It Works](#how-it-works)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [Benchmark Results](#benchmark-results)
- [When to Use Which Engine](#when-to-use-which-engine)
- [Files](#files)
- [Team](#team)
- [License](#license)

---

## Overview

Fast Proximity is a spatial search engine designed for **exact** K-Nearest Neighbor (KNN) queries with a radius constraint. It combines a 1D grid index on the highest-variance dimension with bounding-box filters on secondary dimensions, then delegates to a Numba-JIT compiled kernel for the inner loop — achieving low query latency without approximation.

### v3.0 Improvements over v2.1

| Issue in v2.1 | Fix in v3.0 |
|---|---|
| Fixed buffer `max_c=1000` silently dropped valid candidates | **In-kernel max-heap of size K** — no results ever lost |
| `float32` accumulation caused numerical drift at high D | **`float64` accumulation** inside the distance loop |
| `np.sqrt` called per candidate inside hot loop | **sqrt deferred to post-loop**, once per result |
| Full `np.argsort` O(N log N) on all candidates | **Insertion sort on ≤ K elements** |
| JIT recompiled on every run | **`cache=True`** + automatic warm-up in constructor |
| Fixed `grid_size=5000` regardless of N | **Adaptive default**: `max(500, √N)` |

---

## Features

- ✅ **Exact results** — no approximation, guaranteed K nearest within radius R
- ✅ **JIT-compiled hot path** via [Numba](https://numba.pydata.org/) (`@njit`, `cache=True`, `fastmath=True`)
- ✅ **O(K)** memory for result accumulation (max-heap, no fixed buffer)
- ✅ **Adaptive grid size** scaled to dataset size
- ✅ **Automatic warm-up** on construction — first real query is fast
- ✅ **Correctness verified** against brute-force in the test suite
- ✅ **Pure Python + NumPy + Numba** — no C extensions to compile manually

---

## How It Works

```
Dataset (N × D)
      │
      ▼
  [Build Phase]
  Select top-3 dims by variance
  Build 1D sorted grid on dim[0]         ← O(N log N), done once
      │
      ▼
  [Query Phase]  query_point, R, K
  1. Narrow candidates via grid range on dim[0]    ← O(grid_cells)
  2. Bounding-box filter on dim[1], dim[2]         ← O(grid_hits)
  3. Exact Euclidean distance (float64)            ← O(filtered × D)
  4. Max-heap maintains K best                     ← O(filtered × log K)
  5. Final sort of ≤ K results                     ← O(K log K)
      │
      ▼
  Top-K indices + distances (sorted ascending)
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/your-org/fast-proximity.git
cd fast-proximity

# Install dependencies
pip install numpy scipy scikit-learn numba

# Optional: for benchmark plots
pip install matplotlib
```

> **Python**: 3.9+  
> **Numba**: 0.57+ recommended  
> **NumPy**: 1.23+

---

## Quick Start

```python
import numpy as np
from fast_proximity_v3 import FastProximity

# Build index
np.random.seed(0)
data = np.random.randn(500_000, 8).astype(np.float32)   # 500k points, 8 dims
engine = FastProximity(data)                              # builds index + JIT warm-up

# Query
query = np.random.randn(8).astype(np.float32)
indices, distances = engine.query(query, R=1.5, K=10)

print(indices)    # int64 array, up to K results
print(distances)  # float64 array, Euclidean distances, sorted ascending
```

---

## API Reference

### `FastProximity(data, grid_size=None)`

| Parameter | Type | Description |
|---|---|---|
| `data` | `np.ndarray (N, D)` | Dataset of N points in D dimensions. Converted to `float32` internally. |
| `grid_size` | `int` or `None` | Grid resolution for the primary dimension. Default: `max(500, int(√N))`. Larger values improve filtering but use more memory. |

**Constructor side effects:**
- Selects up to 3 dimensions with highest variance.
- Builds a sorted 1D grid index on the primary dimension.
- Runs a JIT warm-up call so the first real query is fast.

---

### `engine.query(query_point, R, K=5)`

| Parameter | Type | Description |
|---|---|---|
| `query_point` | `np.ndarray (D,)` | The query point. Converted to `float32` internally. |
| `R` | `float` | Search radius (must be > 0). |
| `K` | `int` | Number of nearest neighbors to return (must be > 0). |

**Returns:** `(indices, distances)`

| Return | Type | Description |
|---|---|---|
| `indices` | `np.ndarray[int64]` | Indices into `data`, sorted by distance ascending. Length ≤ K. |
| `distances` | `np.ndarray[float64]` | Euclidean distances, sorted ascending. |

> **Note:** If fewer than K points exist within radius R, the returned arrays will have length < K. This is correct behavior — not an error.

---

## Benchmark Results

Run the full benchmark suite yourself:

```bash
python fast_proximity_test.py
```

### Sample results (N=200k, D=8, R=1.5, K=10, 100 queries)

| Engine | Build time | Query p50 | Query p99 |
|---|---|---|---|
| **FastProximity v3** | ~0.3 s | **~0.15 ms** | ~0.4 ms |
| cKDTree (scipy) | ~0.8 s | ~0.35 ms | ~0.7 ms |
| KDTree (sklearn) | ~1.2 s | ~0.50 ms | ~1.0 ms |
| BallTree (sklearn) | ~1.5 s | ~0.40 ms | ~0.8 ms |
| Brute-Force (numpy) | ~0 s | ~180 ms | ~220 ms |

> Results vary by CPU, dataset distribution, and R. Always benchmark on your own data.

### Benchmark Scenarios

| Scenario | What it tests |
|---|---|
| **Vary N** | Scalability with dataset size (10k → 1M) |
| **Vary D** | Curse of dimensionality (D = 2 → 64) |
| **Vary R** | Impact of search radius on throughput |
| **Correctness** | All engines vs brute-force ground truth |

---

## When to Use Which Engine

```
if D <= 10 and N >= 50_000 and many queries:
    → FastProximity v3  (fastest after warm-up, exact, radius-constrained)
    → cKDTree (scipy)   (no warm-up cost, reliable, great all-rounder)

elif 10 < D <= 30:
    → BallTree (sklearn)  (handles moderate dimensionality better than KDTree)

elif D > 30:
    → Approximate methods: FAISS, Annoy, ScaNN
      (exact methods degrade — "curse of dimensionality")

elif N < 5_000 or one-shot query:
    → Brute-Force  (zero build cost, always correct)
```

### Decision Table

| Criterion | FastProximity v3 | cKDTree | BallTree | Brute-Force |
|---|:---:|:---:|:---:|:---:|
| D ≤ 10 | ✅ Best | ✅ Great | ✅ Good | ⚠️ Slow for N>10k |
| D 10–30 | ⚠️ Degrades | ⚠️ Degrades | ✅ Best | ⚠️ Slow |
| D > 30 | ❌ | ❌ | ⚠️ | ✅ Baseline |
| Large N (>500k) | ✅ | ✅ | ✅ | ❌ |
| Small N (<5k) | ⚠️ Overkill | ⚠️ | ⚠️ | ✅ |
| Exact results | ✅ | ✅ | ✅ | ✅ |
| No JIT/Numba dep | ❌ | ✅ | ✅ | ✅ |
| Warm-up required | ✅ (auto) | ❌ | ❌ | ❌ |
| scikit-learn pipeline | ❌ | ❌ | ✅ | ❌ |

---

## Files

```
fast-proximity/
├── fast_proximity_v3.py      # Core engine — FastProximity class + Numba kernel
├── fast_proximity_test.py    # Full benchmark suite vs cKDTree, KDTree, BallTree
└── README.md                 # This file
```

### `fast_proximity_v3.py`
The engine itself. Contains:
- `_elastic_query_core()` — Numba JIT kernel with max-heap, float64 distances
- `FastProximity` — Python class with adaptive grid index and warm-up

### `fast_proximity_test.py`
Benchmark and validation suite. Contains:
- Wrapper classes for uniform interface across all engines
- 4 benchmark scenarios (vary N, D, R, correctness)
- Decision guide printed at the end

---

## Team

**SLRM Team** — Alex · Gemini · ChatGPT · Claude · Grok · Meta AI

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Contributing

Pull requests welcome. When adding features, please:
1. Add a corresponding test in `fast_proximity_test.py`
2. Verify correctness against brute-force for at least N=50k, D=8
3. Keep the Numba kernel free of Python objects (no lists, no dicts)
 
