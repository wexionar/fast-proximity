"""
================================================================================
FAST PROXIMITY v1.0
High-performance, JIT-accelerated N-Dimensional spatial search engine.
Optimized for ultra-low latency neighbor queries and large-scale datasets.
================================================================================

SLRM Team: Alex · Gemini · ChatGPT · Claude · Grok · Meta AI
Version: 1.0
License: MIT

================================================================================
DESCRIPTION
================================================================================

Fast-Proximity implements a hybrid Spatial Grid Indexing approach combined with 
JIT (Just-In-Time) compilation. It is designed to provide near-native execution 
speeds for proximity searches in Python, making it ideal for real-time 
simulations, robotics, and dynamic data processing where performance 
and flexibility are critical.

================================================================================
"""

import numpy as np
import time
from numba import njit, float32, int64, intp

@njit(fastmath=True)
def _numba_nd_query(query_p, R, gs, cell_starts, cell_ends, data_sorted, sort_idx, K):
    """
    Core JIT-compiled search logic. 
    Maintains near-native speed by avoiding Python's overhead.
    """
    D = query_p.shape[0]
    r_sq = float32(R * R)
    
    # Spatial partitioning logic
    x0 = int(max(0, (query_p[0] - R) * gs))
    x1 = int(min(gs - 1, (query_p[0] + R) * gs))
    y0 = int(max(0, (query_p[1] - R) * gs))
    y1 = int(min(gs - 1, (query_p[1] + R) * gs))
    
    res_indices = []
    res_dists = []
    
    for y in range(y0, y1 + 1):
        for x in range(x0, x1 + 1):
            cid = x + y * gs
            start = cell_starts[cid]
            end = cell_ends[cid]
            
            for i in range(start, end):
                dist_sq = 0.0
                for d in range(D):
                    diff = data_sorted[i, d] - query_p[d]
                    dist_sq += diff * diff
                
                if dist_sq <= r_sq:
                    res_indices.append(sort_idx[i])
                    res_dists.append(np.sqrt(dist_sq))
    
    if len(res_indices) == 0:
        return np.zeros(0, dtype=int64), np.zeros(0, dtype=float32)
    
    indices_arr = np.array(res_indices, dtype=int64)
    dists_arr = np.array(res_dists, dtype=float32)
    
    sort_idx_final = np.argsort(dists_arr)
    return indices_arr[sort_idx_final][:K], dists_arr[sort_idx_final][:K]

class FastProximity:
    """
    Fast-Proximity Engine
    High-speed neighbor search for static and dynamic datasets.
    """
    def __init__(self, data, grid_size=400):
        self.data = data.astype(np.float32)
        self.N, self.D = data.shape
        self.grid_size = grid_size

        # Pre-processing: Grid indexing (O(N log N))
        ix = np.clip((self.data[:, 0] * grid_size).astype(np.int32), 0, grid_size - 1)
        iy = np.clip((self.data[:, 1] * grid_size).astype(np.int32), 0, grid_size - 1)
        self.cell_ids = ix + iy * grid_size

        self.sort_idx = np.argsort(self.cell_ids).astype(np.int64)
        self.data_sorted = self.data[self.sort_idx]
        self.cell_ids_sorted = self.cell_ids[self.sort_idx]

        self.cell_starts = np.searchsorted(self.cell_ids_sorted, np.arange(grid_size**2)).astype(np.intp)
        self.cell_ends = np.append(self.cell_starts[1:], np.intp(self.N))

    def query(self, query_point, R, K=5):
        """
        Returns the K nearest neighbors within radius R.
        """
        return _numba_nd_query(
            query_point.astype(np.float32), 
            float(R), 
            int(self.grid_size), 
            self.cell_starts, 
            self.cell_ends, 
            self.data_sorted, 
            self.sort_idx, 
            int(K)
        )
 
