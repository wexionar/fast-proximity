# 🚀 Fast-Proximity v2.0

**High-performance, JIT-accelerated N-Dimensional spatial search engine for Python.**

Fast-Proximity is a hybrid spatial indexing engine designed for ultra-low latency neighbor queries. By combining **Numba JIT compilation** with an **Adaptive Triple-Filter** logic, it bridges the gap between Python's flexibility and C++ execution speeds.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![Numba](https://img.shields.io/badge/powered%20by-Numba-orange.svg)](https://numba.pydata.org/)

---

## 🧠 Why Fast-Proximity?

Traditional spatial structures like `cKDTree` are excellent for static datasets but can be slow to build in dynamic environments. **Fast-Proximity v2.0** is optimized for the "Build-Query-Update" cycle, making it ideal for:
* **Real-time SLAM & Robotics**
* **Dynamic Particle Simulations**
* **High-frequency Financial Data Clustering**
* **AI/ML Inference** where neighbor lookup is the bottleneck.

## ✨ Key Features in v2.0

* **Adaptive Variance Selection:** The engine automatically analyzes your dataset's variance to choose the most informative dimensions for indexing, avoiding "dead" or collapsed dimensions.
* **Triple-Filter Architecture:** Uses a primary Active Grid combined with two Passive Range Filters to discard up to 99% of candidates before performing expensive Euclidean calculations.
* **Zero-Overhead Numba JIT:** Core search logic is compiled to machine code. Version 2.0 eliminates dynamic list overhead by using pre-allocated memory buffers.
* **Self-Normalizing:** No need to pre-scale your data. The engine handles any coordinate range (GPS, pixels, normalized) automatically.

---

## 📊 Performance Benchmark

*Tested on 1,000,000 points (10 Dimensions) - Average Query Time:*

| Engine | Build Time (ms) | Query Time (ms) | Total (1st Run) |
| :--- | :--- | :--- | :--- |
| `scipy.spatial.cKDTree` | ~880 ms | **~0.25 ms** | ~880.25 ms |
| **Fast-Proximity v2.0** | **~220 ms** | ~2.30 ms | **~222.30 ms** |

> **Verdict:** Fast-Proximity is **4x faster to build** than a KDTree. It is the superior choice for applications where data changes frequently or where the combined "Build + Query" time is the critical factor.

---

## 🛠️ Installation

```bash
pip install numpy numba
```

## 🚀 Quick Start

```python
import numpy as np
from fast_proximity import FastProximity

# Generate 1 million 10D points
data = np.random.rand(1000000, 10).astype(np.float32)

# Initialize the engine (Automatic variance analysis)
engine = FastProximity(data, grid_size=5000)

# Search for 5 neighbors within radius 0.05
query_point = np.random.rand(10).astype(np.float32)
indices, distances = engine.query(query_point, R=0.05, K=5)

print(f"Found {len(indices)} neighbors!")
```

---

## 👥 SLRM Team & Contributors
Developed by the **SLRM Team**: 
`Alex` · `Gemini` · `ChatGPT` · `Claude` · `Grok` · `Meta AI`

## 📄 License
This project is licensed under the **MIT License**.
 
