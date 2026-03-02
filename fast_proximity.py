"""
================================================================================
FAST PROXIMITY v3.0 - "Correctness + Performance"
High-performance, JIT-accelerated N-Dimensional spatial search engine.
Optimized for ultra-low latency neighbor queries and large-scale datasets.

SLRM Team: Alex · Gemini · ChatGPT · Claude · Grok · Meta AI
Version: 3.0
License: MIT
================================================================================

Improvements over v2.1:
  - [CORRECTNESS] Eliminated fixed buffer max_c=1000; heap of size K is used
  - [CORRECTNESS] Internal distances in float64 to avoid numerical error
  - [PERFORMANCE] Partial argsort instead of full argsort
  - [PERFORMANCE] sqrt only at the end (not inside the loop per candidate)
  - [PERFORMANCE] cache=True in JIT to avoid recompilation between sessions
  - [PERFORMANCE] Automatic warm-up in constructor
  - [ROBUSTNESS]  Handling of edge cases: K > candidates, R=0, D=1
================================================================================
"""

import numpy as np
from numba import njit, int64, float64
import heapq


@njit(cache=True, fastmath=True)
def _elastic_query_core(
    data,           # float32 [N, D]
    query_p,        # float32 [D]
    r_sq,           # float64 — radius squared
    gs,             # int — grid_size
    starts1,        # intp [gs]
    ends1,          # intp [gs]
    s_idx1,         # int64 [N]
    dim1,           # int — primary dimension
    min1,           # float32
    scale1,         # float32
    passive_dims,   # int64 [P]
    passive_ranges, # float32 [P, 2]
    K,              # int
):
    """
    JIT search. Returns up to K neighbors within radius R,
    sorted by ascending Euclidean distance.
    
    Uses a max-heap of size K to avoid fixed buffer.
    """
    N, D = data.shape
    r_val = r_sq ** 0.5

    # --- Range in the primary dimension (normalized) ---
    q1_n = (query_p[dim1] - min1) / scale1
    r1_n = r_val / scale1

    x0 = int(max(0.0, (q1_n - r1_n) * gs))
    x1 = int(min(float(gs - 1), (q1_n + r1_n) * gs))

    num_passive = len(passive_dims)

    # Max-heap emulated with parallel arrays (Numba does not support heapq)
    # We save (-dist_sq, idx) → the largest dist_sq is in heap[0]
    heap_dist = np.empty(K, dtype=float64)
    heap_idx  = np.empty(K, dtype=int64)
    heap_size = 0

    for x in range(x0, x1 + 1):
        for ii in range(starts1[x], ends1[x]):
            idx = s_idx1[ii]

            # Passive filters by bounding-box (fast discard)
            passed = True
            for p in range(num_passive):
                v = data[idx, passive_dims[p]]
                if v < passive_ranges[p, 0] or v > passive_ranges[p, 1]:
                    passed = False
                    break
            if not passed:
                continue

            # Complete Euclidean distance in float64
            dist_sq = 0.0
            for d in range(D):
                diff = float(data[idx, d]) - float(query_p[d])
                dist_sq += diff * diff

            if dist_sq > r_sq:
                continue

            # Maintain heap of K best (max-heap by dist_sq)
            if heap_size < K:
                heap_dist[heap_size] = dist_sq
                heap_idx[heap_size]  = idx
                heap_size += 1
                # Manual sift-up
                pos = heap_size - 1
                while pos > 0:
                    parent = (pos - 1) // 2
                    if heap_dist[pos] > heap_dist[parent]:
                        heap_dist[pos], heap_dist[parent] = heap_dist[parent], heap_dist[pos]
                        heap_idx[pos],  heap_idx[parent]  = heap_idx[parent],  heap_idx[pos]
                        pos = parent
                    else:
                        break
            elif dist_sq < heap_dist[0]:
                # Replace the worst element and sift-down
                heap_dist[0] = dist_sq
                heap_idx[0]  = idx
                pos = 0
                while True:
                    left  = 2 * pos + 1
                    right = 2 * pos + 2
                    largest = pos
                    if left < heap_size and heap_dist[left] > heap_dist[largest]:
                        largest = left
                    if right < heap_size and heap_dist[right] > heap_dist[largest]:
                        largest = right
                    if largest == pos:
                        break
                    heap_dist[pos], heap_dist[largest] = heap_dist[largest], heap_dist[pos]
                    heap_idx[pos],  heap_idx[largest]  = heap_idx[largest],  heap_idx[pos]
                    pos = largest

    if heap_size == 0:
        return np.zeros(0, dtype=int64), np.zeros(0, dtype=float64)

    # Sort the heap_size results by ascending distance (partial sort)
    result_dist = heap_dist[:heap_size].copy()
    result_idx  = heap_idx[:heap_size].copy()

    # Insertion sort over heap_size elements (usually small, K ≤ 100)
    for i in range(1, heap_size):
        key_d = result_dist[i]
        key_i = result_idx[i]
        j = i - 1
        while j >= 0 and result_dist[j] > key_d:
            result_dist[j + 1] = result_dist[j]
            result_idx[j + 1]  = result_idx[j]
            j -= 1
        result_dist[j + 1] = key_d
        result_idx[j + 1]  = key_i

    # sqrt only here (once, outside the hot loop)
    for i in range(heap_size):
        result_dist[i] = result_dist[i] ** 0.5

    return result_idx, result_dist


class FastProximity:
    """
    KNN search engine with 1D grid indexing + passive filters by bbox.

    Parameters
    ----------
    data : np.ndarray, shape (N, D)
        Dataset of points.
    grid_size : int
        Grid resolution. Increasing improves filtering but uses more memory.
        Recommended: max(500, int(N ** 0.5))
    """

    def __init__(self, data: np.ndarray, grid_size: int = None):
        self.data = data.astype(np.float32)
        self.N, self.D = self.data.shape

        if grid_size is None:
            grid_size = max(500, int(self.N ** 0.5))
        self.grid_size = grid_size

        # Select up to 3 dimensions of highest variance
        variances = np.var(self.data, axis=0)
        n_dims = min(self.D, 3)
        self.dims = np.argsort(variances)[::-1][:n_dims]

        # Build 1D grid in the primary dimension
        d1 = self.dims[0]
        self.m1 = float(self.data[:, d1].min())
        max1 = float(self.data[:, d1].max())
        self.s1 = float(max1 - self.m1) if max1 > self.m1 else 1.0

        c_ids1 = np.clip(
            ((self.data[:, d1] - self.m1) / self.s1 * grid_size).astype(np.int32),
            0, grid_size - 1
        )
        self.s_idx1  = np.argsort(c_ids1).astype(np.int64)
        ids1_sorted  = c_ids1[self.s_idx1]

        self.starts1 = np.searchsorted(ids1_sorted, np.arange(grid_size)).astype(np.intp)
        self.ends1   = np.append(self.starts1[1:], np.intp(self.N))

        # Warm-up: compile the JIT kernel with dummy data
        self._warmup()

    def _warmup(self):
        dummy_data  = np.zeros((2, self.D), dtype=np.float32)
        dummy_query = np.zeros(self.D, dtype=np.float32)
        dummy_pd    = self.dims[1:].astype(np.int64)
        dummy_pr    = np.zeros((len(dummy_pd), 2), dtype=np.float32)
        _elastic_query_core(
            dummy_data, dummy_query, np.float64(1.0),
            2,
            np.array([0], dtype=np.intp), np.array([1], dtype=np.intp),
            np.array([0, 1], dtype=np.int64),
            int(self.dims[0]), np.float32(0.0), np.float32(1.0),
            dummy_pd, dummy_pr, 1
        )

    def query(self, query_point: np.ndarray, R: float, K: int = 5):
        """
        Searches for the K nearest neighbors within radius R.

        Parameters
        ----------
        query_point : np.ndarray, shape (D,)
        R           : float — search radius
        K           : int   — number of desired neighbors

        Returns
        -------
        indices   : np.ndarray int64, shape (≤K,)
        distances : np.ndarray float64, shape (≤K,)
            Sorted by ascending distance.
        """
        if R <= 0:
            raise ValueError(f"R must be > 0, received: {R}")
        if K <= 0:
            raise ValueError(f"K must be > 0, received: {K}")

        qp = query_point.astype(np.float32)

        passive_dims   = self.dims[1:].astype(np.int64)
        passive_ranges = np.empty((len(passive_dims), 2), dtype=np.float32)
        for i, d in enumerate(passive_dims):
            passive_ranges[i, 0] = qp[d] - R
            passive_ranges[i, 1] = qp[d] + R

        return _elastic_query_core(
            self.data,
            qp,
            np.float64(R * R),
            int(self.grid_size),
            self.starts1,
            self.ends1,
            self.s_idx1,
            int(self.dims[0]),
            np.float32(self.m1),
            np.float32(self.s1),
            passive_dims,
            passive_ranges,
            int(K),
        )


# ---------------------------------------------------------------------------
# Example of use / smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import time

    np.random.seed(42)
    N, D = 500_000, 16
    data = np.random.randn(N, D).astype(np.float32)

    print(f"Building index for N={N}, D={D}...")
    t0 = time.perf_counter()
    engine = FastProximity(data)
    print(f"  Index ready in {time.perf_counter() - t0:.3f}s (includes JIT warm-up)")

    query = np.random.randn(D).astype(np.float32)
    R, K = 3.0, 10

    # Benchmark
    times = []
    for _ in range(200):
        q = np.random.randn(D).astype(np.float32)
        t0 = time.perf_counter()
        idx, dist = engine.query(q, R, K)
        times.append(time.perf_counter() - t0)

    print(f"\nBenchmark (200 queries, R={R}, K={K}):")
    print(f"  Mean:   {np.mean(times)*1000:.3f} ms")
    print(f"  Median: {np.median(times)*1000:.3f} ms")
    print(f"  P99:    {np.percentile(times, 99)*1000:.3f} ms")

    # Correctness verification against brute-force
    print("\nVerifying correctness vs brute-force...")
    q = np.random.randn(D).astype(np.float32)
    idx_fast, dist_fast = engine.query(q, R, K)

    diffs = data - q
    dists_bf = np.sqrt((diffs * diffs).sum(axis=1))
    mask = dists_bf <= R
    bf_idx = np.where(mask)[0]
    bf_sorted = bf_idx[np.argsort(dists_bf[bf_idx])][:K]

    print(f"  Found by engine:      {len(idx_fast)}")
    print(f"  Found by brute-force: {len(bf_sorted)}")
    matches = set(idx_fast.tolist()) == set(bf_sorted.tolist())
    print(f"  Identical results: {'✅ YES' if matches else '❌ NO — check!'}")
 
