"""
================================================================================
FAST PROXIMITY v6.0 (C++) — Python Benchmark & Comparison Suite
================================================================================
Compares Fast Proximity v6.0 (C++ binary) against pure-Python KNN engines:
  - scipy.spatial.cKDTree
  - sklearn.neighbors.KDTree
  - sklearn.neighbors.BallTree
  - Brute-force numpy baseline

Metrics:
  - Index build time
  - Query latency: mean, p50, p95, p99
  - Correctness vs brute-force ground truth
  - Scalability across N, D, R

NOTE: Fast Proximity v6.0 must be compiled first:
  g++ -O3 -march=native -funroll-loops -std=c++17 -o fp6 fast_proximity.cpp
  Place the binary (fp6 / fp6.exe) in the same directory as this script.
================================================================================
"""

import time
import subprocess
import os
import sys
import warnings
import numpy as np
from scipy.spatial import cKDTree
from sklearn.neighbors import KDTree, BallTree

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def timer_fn(fn, *args, reps=5, **kwargs):
    """Median time over `reps` runs."""
    times = []
    result = None
    for _ in range(reps):
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        times.append(time.perf_counter() - t0)
    return float(np.median(times)), result


def brute_force_knn(data, query, R, K):
    diffs = data - query
    dists = np.sqrt((diffs * diffs).sum(axis=1))
    mask  = dists <= R
    cands = np.where(mask)[0]
    if len(cands) == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float64)
    order = np.argsort(dists[cands])
    top   = cands[order][:K]
    return top, dists[top]


def check_correctness(ids_engine, ids_gt):
    return set(ids_engine.tolist()) == set(ids_gt.tolist())


def print_header(title):
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)


def print_table(headers, rows, col_width=20):
    fmt = ("{{:<{}}}".format(col_width)) * len(headers)
    print(fmt.format(*headers))
    print("-" * (col_width * len(headers)))
    for row in rows:
        print(fmt.format(*[str(c) for c in row]))


# ---------------------------------------------------------------------------
# Engine wrappers — uniform interface: build(data) / query(q, R, K)
# ---------------------------------------------------------------------------

class CKDTreeEngine:
    name = "cKDTree (scipy)"
    def __init__(self, data): self.data = data; self.tree = cKDTree(data)
    def query(self, q, R, K):
        idx = self.tree.query_ball_point(q, R)
        if not idx: return np.array([], dtype=np.int64), np.array([], dtype=np.float64)
        idx = np.array(idx, dtype=np.int64)
        d   = np.sqrt(((self.data[idx] - q)**2).sum(axis=1))
        o   = np.argsort(d)[:K]
        return idx[o], d[o]


class SKLKDTreeEngine:
    name = "KDTree (sklearn)"
    def __init__(self, data): self.data = data; self.tree = KDTree(data)
    def query(self, q, R, K):
        idx = self.tree.query_radius(q.reshape(1,-1), R)[0]
        if not len(idx): return np.array([], dtype=np.int64), np.array([], dtype=np.float64)
        d = np.sqrt(((self.data[idx] - q)**2).sum(axis=1))
        o = np.argsort(d)[:K]
        return idx[o], d[o]


class BallTreeEngine:
    name = "BallTree (sklearn)"
    def __init__(self, data): self.data = data; self.tree = BallTree(data)
    def query(self, q, R, K):
        idx = self.tree.query_radius(q.reshape(1,-1), R)[0]
        if not len(idx): return np.array([], dtype=np.int64), np.array([], dtype=np.float64)
        d = np.sqrt(((self.data[idx] - q)**2).sum(axis=1))
        o = np.argsort(d)[:K]
        return idx[o], d[o]


class BruteForceEngine:
    name = "Brute-Force (numpy)"
    def __init__(self, data): self.data = data
    def query(self, q, R, K): return brute_force_knn(self.data, q, R, K)


# ---------------------------------------------------------------------------
# Fast Proximity v6.0 timing via subprocess
# (the C++ binary runs its own internal benchmark and prints results)
# ---------------------------------------------------------------------------

FP6_BINARY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fp6")

def run_fp6_benchmark():
    """Run the compiled C++ binary and parse its output."""
    if not os.path.isfile(FP6_BINARY):
        print(f"\n  [SKIP] Fast Proximity v6.0 binary not found at: {FP6_BINARY}")
        print(  "         Compile with:")
        print(  "         g++ -O3 -march=native -funroll-loops -std=c++17 -o fp6 fast_proximity.cpp\n")
        return None
    result = subprocess.run([FP6_BINARY], capture_output=True, text=True)
    return result.stdout


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------

def build_engines(data, engine_classes):
    built = []
    for cls in engine_classes:
        t0 = time.perf_counter()
        eng = cls(data)
        built.append((eng, time.perf_counter() - t0))
    return built


def query_latencies(engines_built, queries, R, K, reps=5):
    stats = []
    for eng, build_t in engines_built:
        times = []
        for _ in range(reps):
            for q in queries:
                t0 = time.perf_counter()
                eng.query(q, R, K)
                times.append(time.perf_counter() - t0)
        times = sorted(times)
        n = len(times)
        stats.append({
            "name":      eng.name,
            "build_ms":  build_t * 1000,
            "mean_ms":   np.mean(times) * 1000,
            "p50_ms":    np.percentile(times, 50) * 1000,
            "p95_ms":    np.percentile(times, 95) * 1000,
            "p99_ms":    np.percentile(times, 99) * 1000,
        })
    return stats


def correctness_check(engines_built, queries, R, K, data, n_check=30):
    print(f"\n  Correctness vs brute-force ({n_check} queries):")
    for eng, _ in engines_built:
        ok = 0
        for q in queries[:n_check]:
            gt_idx, _ = brute_force_knn(data, q, R, K)
            fp_idx, _ = eng.query(q, R, K)
            if check_correctness(fp_idx, gt_idx): ok += 1
        status = "✅ PASS" if ok == n_check else f"❌ FAIL ({ok}/{n_check})"
        print(f"    {eng.name:<28} {status}")


def print_stats(stats):
    headers = ["Engine", "Build (ms)", "Mean (ms)", "p50 (ms)", "p95 (ms)", "p99 (ms)"]
    rows = [(s["name"],
             f"{s['build_ms']:.1f}",
             f"{s['mean_ms']:.3f}",
             f"{s['p50_ms']:.3f}",
             f"{s['p95_ms']:.3f}",
             f"{s['p99_ms']:.3f}") for s in stats]
    print_table(headers, rows, col_width=22)


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

ALL_ENGINES = [CKDTreeEngine, SKLKDTreeEngine, BallTreeEngine, BruteForceEngine]
FAST_ENGINES = [CKDTreeEngine, SKLKDTreeEngine, BallTreeEngine]  # skip BF for large N


def scenario_vary_N():
    print_header("SCENARIO 1 — Varying N  |  D=5, R=0.08, K=10")
    D, R, K = 5, 0.08, 10
    Ns = [10_000, 50_000, 200_000, 500_000]

    for N in Ns:
        print(f"\n  N = {N:,}")
        np.random.seed(0)
        data    = np.random.rand(N, D).astype(np.float32)
        queries = np.random.rand(100, D).astype(np.float32)
        classes = ALL_ENGINES if N <= 50_000 else FAST_ENGINES
        built   = build_engines(data, classes)
        stats   = query_latencies(built, queries, R, K, reps=3)
        print_stats(stats)


def scenario_vary_D():
    print_header("SCENARIO 2 — Varying D (curse of dimensionality)  |  N=100k, R=2.5, K=10")
    N, R, K = 100_000, 2.5, 10
    Ds = [2, 4, 8, 16, 32]

    for D in Ds:
        print(f"\n  D = {D}")
        np.random.seed(1)
        data    = np.random.randn(N, D).astype(np.float32)
        queries = np.random.randn(80, D).astype(np.float32)
        built   = build_engines(data, FAST_ENGINES)
        stats   = query_latencies(built, queries, R, K, reps=3)
        print_stats(stats)


def scenario_vary_R():
    print_header("SCENARIO 3 — Varying R  |  N=200k, D=5, K=10")
    N, D, K = 200_000, 5, 10
    Rs = [0.02, 0.05, 0.1, 0.2, 0.4]

    np.random.seed(2)
    data    = np.random.rand(N, D).astype(np.float32)
    queries = np.random.rand(80, D).astype(np.float32)

    for R in Rs:
        print(f"\n  R = {R}")
        built = build_engines(data, FAST_ENGINES)
        stats = query_latencies(built, queries, R, K, reps=3)
        print_stats(stats)


def scenario_correctness():
    print_header("SCENARIO 4 — Correctness Validation  |  N=30k, D=5, R=0.1, K=10")
    N, D, R, K = 30_000, 5, 0.1, 10
    np.random.seed(42)
    data    = np.random.rand(N, D).astype(np.float32)
    queries = np.random.rand(50, D).astype(np.float32)
    built   = build_engines(data, FAST_ENGINES)
    correctness_check(built, queries, R, K, data)


def scenario_fp6():
    print_header("FAST PROXIMITY v6.0 (C++) — Internal Benchmark Output")
    output = run_fp6_benchmark()
    if output:
        for line in output.strip().splitlines():
            print(f"  {line}")


def print_decision_guide():
    print_header("DECISION GUIDE — When to Use Which Engine")
    print("""
  ┌──────────────────────────────────────────────────────────────────────┐
  │  Scenario                           Recommended Engine               │
  ├──────────────────────────────────────────────────────────────────────┤
  │  D ≤ 10, N ≥ 100k, many queries    Fast Proximity v6.0 (C++)        │
  │  D ≤ 10, no C++ toolchain          cKDTree (scipy)  ← best fallback │
  │  D 10–30, general use              BallTree (sklearn)                │
  │  D > 30                            FAISS / Annoy (approximate)       │
  │  N < 5k or one-shot                Brute-Force (zero build cost)     │
  │  scikit-learn pipeline             KDTree / BallTree                 │
  └──────────────────────────────────────────────────────────────────────┘

  Key trade-offs:
  ┌─────────────────────┬──────────┬───────────┬──────────┬─────────────┐
  │                     │ FP v6.0  │ cKDTree   │ BallTree │ Brute-Force │
  ├─────────────────────┼──────────┼───────────┼──────────┼─────────────┤
  │ D ≤ 10              │ ✅ Best  │ ✅ Great  │ ✅ Good  │ ⚠️ slow N>10k│
  │ D 10–30             │ ⚠️       │ ⚠️        │ ✅ Best  │ ⚠️          │
  │ D > 30              │ ❌       │ ❌        │ ⚠️       │ ✅ baseline  │
  │ Large N (>500k)     │ ✅       │ ✅        │ ✅       │ ❌          │
  │ Exact results       │ ✅       │ ✅        │ ✅       │ ✅          │
  │ No compile step     │ ❌       │ ✅        │ ✅       │ ✅          │
  │ Native speed (C++)  │ ✅       │ ✅        │ ❌       │ ❌          │
  └─────────────────────┴──────────┴───────────┴──────────┴─────────────┘
""")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n" + "=" * 72)
    print("  Fast Proximity v6.0 — Full Benchmark Suite")
    print("  Engines: FP v6.0 (C++), cKDTree, KDTree, BallTree, Brute-Force")
    print("=" * 72)

    scenario_fp6()
    scenario_vary_N()
    scenario_vary_D()
    scenario_vary_R()
    scenario_correctness()
    print_decision_guide()

    print("\n" + "=" * 72)
    print("  Benchmark complete.")
    print("=" * 72 + "\n")
 
