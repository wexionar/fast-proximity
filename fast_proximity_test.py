"""
================================================================================
FAST PROXIMITY v3.0 — Benchmark & Comparison Suite
================================================================================
Compares FastProximity v3.0 against:
  - scipy.spatial.cKDTree
  - sklearn.neighbors.KDTree
  - sklearn.neighbors.BallTree
  - Brute-force (numpy baseline)

Metrics measured:
  - Index build time
  - Query time (single + batch)
  - Result correctness (vs brute-force ground truth)
  - Scalability across N, D, R, K

Usage:
  python fast_proximity_test.py

Requirements:
  numpy, scipy, scikit-learn, numba, matplotlib (optional for plots)
================================================================================
"""

import time
import warnings
import numpy as np
from scipy.spatial import cKDTree
from sklearn.neighbors import KDTree, BallTree

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Try importing FastProximity — adjust path if needed
# ---------------------------------------------------------------------------
try:
    from fast_proximity import FastProximity
    HAS_FP = True
except ImportError:
    print("[WARNING] fast_proximity.py not found in path. Skipping FastProximity.")
    HAS_FP = False


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def timer(fn, *args, reps=10, **kwargs):
    """Run fn(*args, **kwargs) `reps` times and return (mean_sec, last_result)."""
    result = None
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        times.append(time.perf_counter() - t0)
    return float(np.median(times)), result


def brute_force_knn(data, query, R, K):
    """Ground-truth KNN via numpy brute-force."""
    diffs = data - query
    dists = np.sqrt((diffs * diffs).sum(axis=1))
    mask = dists <= R
    candidates = np.where(mask)[0]
    if len(candidates) == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float64)
    order = np.argsort(dists[candidates])
    top = candidates[order][:K]
    return top, dists[top]


def check_correctness(idx_engine, idx_gt):
    """Return True if both result sets are identical (order-insensitive)."""
    return set(idx_engine.tolist()) == set(idx_gt.tolist())


def print_header(title):
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)


def print_table(headers, rows, col_width=18):
    fmt = ("{{:<{}}}".format(col_width)) * len(headers)
    print(fmt.format(*headers))
    print("-" * (col_width * len(headers)))
    for row in rows:
        print(fmt.format(*[str(x) for x in row]))


# ---------------------------------------------------------------------------
# Engine wrappers  (uniform interface)
# ---------------------------------------------------------------------------

class CKDTreeEngine:
    name = "cKDTree (scipy)"

    def __init__(self, data, **kw):
        self.data = data
        self.tree = cKDTree(data, leafsize=kw.get("leafsize", 16))

    def query(self, q, R, K):
        idx = self.tree.query_ball_point(q, R)
        if not idx:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float64)
        idx = np.array(idx, dtype=np.int64)
        dists = np.sqrt(((self.data[idx] - q) ** 2).sum(axis=1))
        order = np.argsort(dists)[:K]
        return idx[order], dists[order]


class SKLKDTreeEngine:
    name = "KDTree (sklearn)"

    def __init__(self, data, **kw):
        self.data = data
        self.tree = KDTree(data, leaf_size=kw.get("leaf_size", 40))

    def query(self, q, R, K):
        idx = self.tree.query_radius(q.reshape(1, -1), R)[0]
        if len(idx) == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float64)
        dists = np.sqrt(((self.data[idx] - q) ** 2).sum(axis=1))
        order = np.argsort(dists)[:K]
        return idx[order], dists[order]


class BallTreeEngine:
    name = "BallTree (sklearn)"

    def __init__(self, data, **kw):
        self.data = data
        self.tree = BallTree(data, leaf_size=kw.get("leaf_size", 40))

    def query(self, q, R, K):
        idx = self.tree.query_radius(q.reshape(1, -1), R)[0]
        if len(idx) == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float64)
        dists = np.sqrt(((self.data[idx] - q) ** 2).sum(axis=1))
        order = np.argsort(dists)[:K]
        return idx[order], dists[order]


class BruteForceEngine:
    name = "Brute-Force (numpy)"

    def __init__(self, data, **kw):
        self.data = data

    def query(self, q, R, K):
        return brute_force_knn(self.data, q, R, K)


class FastProximityEngine:
    name = "FastProximity v3"

    def __init__(self, data, **kw):
        self.engine = FastProximity(data, grid_size=kw.get("grid_size", None))

    def query(self, q, R, K):
        return self.engine.query(q, R, K)


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------

def build_engines(data, engine_classes, **kw):
    """Build all engines and return list of (engine, build_time)."""
    results = []
    for cls in engine_classes:
        t0 = time.perf_counter()
        eng = cls(data, **kw)
        build_t = time.perf_counter() - t0
        results.append((eng, build_t))
    return results


def run_query_benchmark(engines_built, queries, R, K, reps=5):
    """
    For each engine, run all queries `reps` times and collect timing.
    Returns list of dicts with stats.
    """
    stats = []
    for eng, build_t in engines_built:
        times = []
        for _ in range(reps):
            t0 = time.perf_counter()
            for q in queries:
                eng.query(q, R, K)
            times.append((time.perf_counter() - t0) / len(queries))
        stats.append({
            "name":       eng.name,
            "build_ms":   build_t * 1000,
            "q_mean_ms":  np.mean(times) * 1000,
            "q_p50_ms":   np.median(times) * 1000,
            "q_p99_ms":   np.percentile(times, 99) * 1000 if len(times) > 1 else np.nan,
        })
    return stats


def correctness_check(engines_built, queries, R, K, data):
    """Compare each engine's results vs brute-force ground truth."""
    print("\n  Correctness check (first 20 queries):")
    for eng, _ in engines_built:
        ok = 0
        total = min(20, len(queries))
        for q in queries[:total]:
            gt_idx, _ = brute_force_knn(data, q, R, K)
            idx, _    = eng.query(q, R, K)
            if check_correctness(idx, gt_idx):
                ok += 1
        status = "✅ PASS" if ok == total else f"❌ FAIL ({ok}/{total})"
        print(f"    {eng.name:<25} {status}")


def print_stats(stats):
    headers = ["Engine", "Build (ms)", "Query p50 (ms)", "Query p99 (ms)"]
    rows = [
        (
            s["name"],
            f"{s['build_ms']:.1f}",
            f"{s['q_p50_ms']:.3f}",
            f"{s['q_p99_ms']:.3f}",
        )
        for s in stats
    ]
    print_table(headers, rows, col_width=22)


# ---------------------------------------------------------------------------
# Scenario 1 — Varying N (fixed D, R, K)
# ---------------------------------------------------------------------------

def benchmark_vary_N():
    print_header("SCENARIO 1: Varying N  |  D=8, R=1.5, K=10")
    D, R, K = 8, 1.5, 10
    Ns = [10_000, 100_000, 500_000, 1_000_000]
    n_queries = 100

    engine_classes = [FastProximityEngine, CKDTreeEngine, SKLKDTreeEngine, BallTreeEngine, BruteForceEngine] \
                     if HAS_FP else [CKDTreeEngine, SKLKDTreeEngine, BallTreeEngine, BruteForceEngine]

    for N in Ns:
        print(f"\n  N = {N:,}")
        np.random.seed(0)
        data    = np.random.randn(N, D).astype(np.float32)
        queries = np.random.randn(n_queries, D).astype(np.float32)

        skip_bf = N > 200_000
        classes = engine_classes if not skip_bf else [c for c in engine_classes if c != BruteForceEngine]

        engines_built = build_engines(data, classes)
        stats         = run_query_benchmark(engines_built, queries, R, K, reps=3)
        print_stats(stats)


# ---------------------------------------------------------------------------
# Scenario 2 — Varying D (curse of dimensionality)
# ---------------------------------------------------------------------------

def benchmark_vary_D():
    print_header("SCENARIO 2: Varying D  |  N=100k, R=2.5, K=10")
    N, R, K = 100_000, 2.5, 10
    Ds = [2, 4, 8, 16, 32, 64]
    n_queries = 100

    engine_classes = [FastProximityEngine, CKDTreeEngine, SKLKDTreeEngine, BallTreeEngine] \
                     if HAS_FP else [CKDTreeEngine, SKLKDTreeEngine, BallTreeEngine]

    for D in Ds:
        print(f"\n  D = {D}")
        np.random.seed(1)
        data    = np.random.randn(N, D).astype(np.float32)
        queries = np.random.randn(n_queries, D).astype(np.float32)

        engines_built = build_engines(data, engine_classes)
        stats         = run_query_benchmark(engines_built, queries, R, K, reps=3)
        print_stats(stats)


# ---------------------------------------------------------------------------
# Scenario 3 — Varying R (search radius)
# ---------------------------------------------------------------------------

def benchmark_vary_R():
    print_header("SCENARIO 3: Varying R  |  N=200k, D=8, K=10")
    N, D, K = 200_000, 8, 10
    Rs = [0.1, 0.5, 1.0, 2.0, 5.0]
    n_queries = 100

    np.random.seed(2)
    data    = np.random.randn(N, D).astype(np.float32)
    queries = np.random.randn(n_queries, D).astype(np.float32)

    engine_classes = [FastProximityEngine, CKDTreeEngine, BallTreeEngine] \
                     if HAS_FP else [CKDTreeEngine, BallTreeEngine]

    engines_built = build_engines(data, engine_classes)

    headers = ["Engine"] + [f"R={r}" for r in Rs]
    rows    = {cls.name if hasattr(cls, "name") else str(cls): [] for cls in engine_classes}

    for eng, _ in engines_built:
        rows[eng.name] = [eng.name]

    for R in Rs:
        print(f"\n  R = {R}")
        for eng, build_t in engines_built:
            times = []
            for _ in range(3):
                t0 = time.perf_counter()
                for q in queries:
                    eng.query(q, R, K)
                times.append((time.perf_counter() - t0) / len(queries) * 1000)
            rows[eng.name].append(f"{np.median(times):.3f} ms")
            print(f"    {eng.name:<25} {np.median(times):.3f} ms/query")


# ---------------------------------------------------------------------------
# Scenario 4 — Correctness validation
# ---------------------------------------------------------------------------

def benchmark_correctness():
    print_header("SCENARIO 4: Correctness Validation  |  N=50k, D=8, R=1.5, K=10")
    N, D, R, K = 50_000, 8, 1.5, 10

    np.random.seed(42)
    data    = np.random.randn(N, D).astype(np.float32)
    queries = np.random.randn(50, D).astype(np.float32)

    engine_classes = [FastProximityEngine, CKDTreeEngine, SKLKDTreeEngine, BallTreeEngine] \
                     if HAS_FP else [CKDTreeEngine, SKLKDTreeEngine, BallTreeEngine]

    engines_built = build_engines(data, engine_classes)
    correctness_check(engines_built, queries, R, K, data)


# ---------------------------------------------------------------------------
# Scenario 5 — When to use which engine (summary recommendation)
# ---------------------------------------------------------------------------

def print_recommendations():
    print_header("WHEN TO USE WHICH ENGINE — Decision Guide")
    guide = """
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  Use Case                        Recommended Engine                    │
  ├─────────────────────────────────────────────────────────────────────────┤
  │  D ≤ 10, N > 100k, many queries  cKDTree (scipy) ← best all-rounder    │
  │  D ≤ 10, repeated queries, speed FastProximity v3 (after JIT warm-up)  │
  │  D 10–30, general use            BallTree (sklearn)                    │
  │  D > 30 ("curse of dims")        Brute-Force or FAISS/Annoy (approx.)  │
  │  Small dataset N < 5k            Brute-Force (no build overhead)       │
  │  Exact + radius constraint       FastProximity v3 or cKDTree           │
  │  Batch queries, scikit pipeline  KDTree / BallTree (sklearn)           │
  │  GPU / billion-scale             FAISS (not covered here)              │
  └─────────────────────────────────────────────────────────────────────────┘

  Key trade-offs:
  • FastProximity v3:  fastest queries for medium N + low D after JIT warm-up;
                       requires Numba; build time is low; no approximate mode.
  • cKDTree (scipy):   C-optimized, no warm-up cost, reliable across N and D≤15.
  • BallTree:          better than KDTree for D > 10; supports many metrics.
  • KDTree (sklearn):  solid for D ≤ 10; slower build than cKDTree.
  • Brute-Force:       always correct; scales as O(N·D) per query — avoid for N>10k.

  Rule of thumb:
    if D <= 10 and N >= 50_000:     cKDTree or FastProximity v3
    elif 10 < D <= 30:              BallTree
    elif D > 30:                    approximate methods (FAISS, Annoy, ScaNN)
    else (small N or one-shot):     Brute-Force
"""
    print(guide)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "=" * 72)
    print("  FAST PROXIMITY v3.0 — Full Benchmark Suite")
    print("  Engines: FastProximity v3, cKDTree, KDTree, BallTree, Brute-Force")
    print("=" * 72)

    if not HAS_FP:
        print("\n[INFO] Running without FastProximity (import failed).")

    benchmark_vary_N()
    benchmark_vary_D()
    benchmark_vary_R()
    benchmark_correctness()
    print_recommendations()

    print("\n" + "=" * 72)
    print("  Benchmark complete.")
    print("=" * 72 + "\n")
 
