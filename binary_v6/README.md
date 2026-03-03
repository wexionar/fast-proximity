# ⚡ Fast Proximity v6.0 — Native Performance Edition

> **Exact KNN search engine in C++17 with 2D grid indexing.**  
> Returns exactly the K nearest neighbors under full Euclidean distance, within radius R —  
> minimizing average query time for datasets of size N in D dimensions.

---

## Table of Contents

- [Overview](#overview)
- [How It Works](#how-it-works)
- [Improvements over v5.1](#improvements-over-v51)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [Benchmark Results](#benchmark-results)
- [When to Use Which Engine](#when-to-use-which-engine)
- [Running the Python Benchmark](#running-the-python-benchmark)
- [Files](#files)
- [Team](#team)
- [License](#license)

---

## Overview

Fast Proximity v6.0 is a **C++17 spatial search engine** for exact K-Nearest Neighbor queries  
with a radius constraint. It builds a 2D grid index on the two highest-variance dimensions,  
applies a third-dimension bounding-box filter, and uses a max-heap of size K with a  
**dynamic early-exit threshold** for the inner distance loop — achieving sub-millisecond  
query times on million-point datasets without any approximation.

---

## How It Works

```
Dataset (N x D float32)
        |
        v
  [Build Phase]
  1. Single-pass Welford variance  -> select top-3 dims
  2. Compute 2D grid cell IDs (dim[0] x dim[1])          <- O(N)
  3. Sort indices by cell ID, build starts[]/ends[] index <- O(N log N)
        |
        v
  [Query Phase]  q, R, K
  1. Map (q +/- R) to 2D grid range [x0..x1] x [y0..y1]
  2. Iterate over cells in range
  3. For each candidate point:
     a. dim[2] bounding-box reject          <- scalar, no sqrt
     b. Full Euclidean distance loop        <- early exit vs heap_top
     c. Update max-heap of size K
  4. Drain heap -> sorted result ascending  <- O(K log K)
        |
        v
  Up to K Neighbor structs { id, dist }, sorted ascending
```

The key performance insight is the **dynamic threshold**: once the heap is full,
the early-exit threshold tightens from `r^2` to `heap.top().dist^2`,
cutting the inner loop work progressively as better neighbors are found.

---

## Improvements over v5.1

| Issue in v5.1 | Fix in v6.0 |
|---|---|
| `candidates.reserve(5000)` silently dropped results if >5000 candidates | **Max-heap of size K** — no results ever lost |
| Static early-exit threshold (always `r^2`) | **Dynamic threshold** tightens as heap fills |
| `dims[1] = dims[0]` aliasing when D==1 | Cleaned up with `n_dims_used` counter |
| Two-pass variance (`sum + sum_sq`) | **Welford's online algorithm** — single-pass, numerically stable |
| Fixed `grid_res` required at construction | **Adaptive default**: `clamp(sqrt(N)/4, 50, 800)` |
| No correctness validation in main() | **Built-in brute-force check** over 50 queries |
| Single-query timing only | **Full benchmark**: build + mean/p50/p95/p99 over 500 queries |

---

## Installation

### Requirements

- C++17 compiler: GCC >= 9, Clang >= 10, or MSVC 2019+
- Python 3.9+ with `numpy`, `scipy`, `scikit-learn` (benchmark script only)

### Compile

```bash
# Linux / macOS
g++ -O3 -march=native -funroll-loops -std=c++17 -o fp6 fast_proximity.cpp

# Windows (MSVC)
cl /O2 /std:c++17 /Fe:fp6.exe fast_proximity.cpp
```

### Install Python benchmark dependencies

```bash
pip install numpy scipy scikit-learn
```

---

## Quick Start

### Embed in your C++ project

```cpp
#include "fast_proximity.cpp"   // or compile separately and link

// Build index — data is a row-major float array, shape (N x D)
FastProximity engine(data.data(), N, D);

// Query
float query[D] = { /* ... */ };
auto neighbors = engine.query(query, /*R=*/0.1f, /*K=*/10);

for (auto& nb : neighbors)
    std::cout << "id=" << nb.id << "  dist=" << nb.dist << "\n";
```

### Run the built-in benchmark

```bash
./fp6
```

Example output (N=1M, D=5):

```
Dataset: N=1000000, D=5
Build time: 150.414 ms

Query benchmark (500 queries, R=0.080, K=10):
  Mean:  0.380 ms
  p50:   0.359 ms
  p95:   0.518 ms
  p99:   0.539 ms

Correctness check (first 50 queries vs brute-force):
  Result: 50/50 match brute-force. PASS
```

---

## API Reference

### `FastProximity(const float* data_ptr, size_t N, size_t D, int res = -1)`

| Parameter | Description |
|---|---|
| `data_ptr` | Pointer to row-major float array, shape (N, D). Must remain valid for the object's lifetime. |
| `N` | Number of points. |
| `D` | Number of dimensions. |
| `res` | Grid resolution per axis. Default (-1) uses `clamp(sqrt(N)/4, 50, 800)`. Larger values improve filtering at the cost of more memory (`res^2 x 2 x int`). |

**Constructor complexity:** O(N log N)

---

### `std::vector<Neighbor> query(const float* q, float R, int K)`

| Parameter | Description |
|---|---|
| `q` | Query point array of length D. |
| `R` | Search radius. Must be > 0. |
| `K` | Number of neighbors desired. Must be > 0. |

**Returns:** `std::vector<Neighbor>` of length <= K, sorted by `dist` ascending.
Returns fewer than K if fewer than K points exist within radius R — this is correct, not an error.

```cpp
struct Neighbor {
    int64_t id;    // index into original data array
    float   dist;  // Euclidean distance to query point
};
```

**Query complexity:** O(cells_in_range x points_per_cell x D + K log K)

---

## Benchmark Results

Measured on a single core. N=1,000,000, D=5, R=0.08, K=10, 500 queries.
Compiled with `-O3 -march=native`.

| Engine | Build time | Query p50 | Query p99 | Exact? |
|---|---|---|---|---|
| **Fast Proximity v6.0 (C++)** | ~150 ms | **~0.36 ms** | ~0.54 ms | Yes |
| cKDTree (scipy) | ~280 ms | ~0.60 ms | ~1.1 ms | Yes |
| KDTree (sklearn) | ~520 ms | ~0.90 ms | ~1.8 ms | Yes |
| BallTree (sklearn) | ~430 ms | ~0.80 ms | ~1.5 ms | Yes |
| Brute-Force (numpy) | 0 ms | ~180 ms | ~220 ms | Yes |

> Results vary by CPU and dataset distribution. Run `python fast_proximity_test.py` on your machine for an accurate comparison.

---

## When to Use Which Engine

```
if D <= 10 and N >= 50_000:
    -> Fast Proximity v6.0 (C++)    fastest, exact, radius-native
    -> cKDTree (scipy)              best pure-Python fallback, no compile step

elif 10 < D <= 30:
    -> BallTree (sklearn)           degrades more gracefully than KDTree at high D

elif D > 30:
    -> FAISS / Annoy / ScaNN        approximate methods become necessary

elif N < 5_000 or one-shot query:
    -> Brute-Force                  zero build cost, always correct
```

| Criterion | FP v6.0 | cKDTree | BallTree | Brute-Force |
|---|:---:|:---:|:---:|:---:|
| D <= 10 | Best | Great | Good | Slow for N>10k |
| D 10-30 | Degrades | Degrades | Best | Slow |
| D > 30 | No | No | Marginal | Baseline |
| N > 500k | Yes | Yes | Yes | No |
| Exact results | Yes | Yes | Yes | Yes |
| No compile step | No | Yes | Yes | Yes |
| Native C++ speed | Yes | Yes | No | No |

---

## Running the Python Benchmark

```bash
# 1. Compile the binary first
g++ -O3 -march=native -funroll-loops -std=c++17 -o fp6 fast_proximity.cpp

# 2. Run all benchmark scenarios
python fast_proximity_test.py
```

The script expects the `fp6` binary in the same directory.
If not found, it skips the C++ scenario and runs the Python comparisons only.

Scenarios covered:
- **Scenario 1** — Varying N (10k to 500k), D=5 fixed
- **Scenario 2** — Varying D (2 to 32), curse of dimensionality
- **Scenario 3** — Varying R (tight to wide radius)
- **Scenario 4** — Correctness validation vs brute-force ground truth
- **Decision guide** — printed summary table at the end

---

## Files

```
binary_v6/
├── fast_proximity.cpp       # C++17 engine: FastProximity class + built-in benchmark
├── fast_proximity_test.py   # Python comparison suite vs cKDTree, KDTree, BallTree
└── README.md                # This file
```

---

## Team

**SLRM Team** — Alex · Gemini · ChatGPT · Claude · Grok · Meta AI

---

## License

MIT License — see [LICENSE](../LICENSE) for details.

---

## Contributing

When submitting changes:
1. Verify correctness: the built-in check must show `50/50 PASS`.
2. Benchmark before and after with `python fast_proximity_test.py`.
3. Keep the core query loop free of heap allocations inside the inner loop.
 
