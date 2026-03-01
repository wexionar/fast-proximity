# Fast-Proximity 🚀

**Fast-Proximity** is a high-performance, N-Dimensional spatial search engine for Python. It is designed as a lightweight, JIT-compiled engine optimized for ultra-low latency queries and large-scale datasets.

Built on top of **NumPy** and **Numba**, it achieves near-native execution speeds by compiling Python logic directly into machine code.

## Key Features

* **Extreme Performance:** Query 1 million points in ~0.16 ms (2D) or ~0.44 ms (10D).
* **JIT-Accelerated:** Leverages Numba's `@njit` to eliminate Python interpreter overhead.
* **Streamlined Architecture:** Uses a hybrid Spatial Grid Indexing approach for balanced construction and search times.
* **Zero-Weight:** Minimal dependency stack (NumPy + Numba).
* **N-Dimensional:** Efficiently scales from 2D spatial mapping to high-dimensional feature matching.

## Performance Benchmark (1M Points)

| Dimension | Engine | Query Time |
|-----------|--------|------------|
| 2D | Fast-Proximity | **0.16 ms** |
| 10D | Fast-Proximity | **0.44 ms** |

## Quick Start

```python
import numpy as np
from fast_proximity import FastProximity

# 1. Initialize with 1 million 10D points
data = np.random.rand(1000000, 10).astype(np.float32)
engine = FastProximity(data, grid_size=400)

# 2. Query neighbors within radius R
query_p = np.random.rand(10).astype(np.float32)
indices, distances = engine.query(query_p, R=0.1, K=5)

if indices is not None:
    print(f"Found {len(indices)} neighbors in record time!")
```

## Why Fast-Proximity?

While traditional spatial structures are often static and hard to modify, **Fast-Proximity** is built with flexibility in mind. By using a grid-based approach instead of a rigid tree, it provides:

1. **Near-instant initialization.**
2. **High throughput** for real-time applications (Robotics, Physics Simulations, AI).
3. **Full transparency:** Custom distance metrics or search logic can be easily injected into the JIT loop.

## Roadmap
- [ ] **Fast-Triangulation**: Upcoming high-speed triangulation module.
- [ ] Dynamic point insertion and deletion (CRUD support).
- [ ] Support for custom distance metrics (Manhattan, Cosine, etc.).

## License
MIT
