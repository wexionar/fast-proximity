"""
================================================================================
FAST PROXIMITY v2.0 - "The Triple Filter Evolution"
High-performance, JIT-accelerated N-Dimensional spatial search engine.
Optimized for ultra-low latency neighbor queries and large-scale datasets.
================================================================================

SLRM Team: Alex · Gemini · ChatGPT · Claude · Grok · Meta AI
Version: 2.0 (Triple-Filter Smart Engine)
License: MIT

================================================================================
DESCRIPTION
================================================================================

Fast-Proximity v2.0 implements an Adaptive 1D-Grid with Multi-Dimensional 
Passive Filtering. Unlike static grids, this engine analyzes data variance 
to select the most informative dimensions for indexing and filtering.

Key improvements:
- Automatic variance-based dimension selection.
- Dynamic normalization (handles any data scale).
- Pre-allocated Numba buffers (no dynamic list overhead).
- Triple-stage filtering (1 Active Grid + 2 Passive Range Filters).

================================================================================
"""

import numpy as np
from numba import njit, float32, int64, intp

@njit(fastmath=True)
def _numba_triple_filter_query(data, query_p, r_sq, gs, 
                               starts1, ends1, s_idx1, dim1, min1, scale1,
                               dim2, q2_l, q2_h,
                               dim3, q3_l, q3_h, K):
    """
    Core JIT-compiled search logic. 
    Triple-stage filtering minimizes Euclidean distance computations.
    """
    N, D = data.shape
    r_val = np.sqrt(r_sq)
    
    # 1. Active Grid Normalization (Dimension with max variance)
    q1_n = (query_p[dim1] - min1) / scale1
    r1_n = r_val / scale1
    
    # 2. Determine cell range
    x0 = int(max(0, (q1_n - r1_n) * gs))
    x1 = int(min(gs - 1, (q1_n + r1_n) * gs))
    
    # Buffer pre-allocation for performance (no .append)
    max_c = 1000 
    res_idx = np.empty(max_c, dtype=int64)
    res_dst = np.empty(max_c, dtype=float32)
    count = 0

    # 3. Execution Loop
    for x in range(x0, x1 + 1):
        for i in range(starts1[x], ends1[x]):
            idx = s_idx1[i]
            
            # PASSIVE FILTER A (Second best dimension)
            if data[idx, dim2] >= q2_l and data[idx, dim2] <= q2_h:
                # PASSIVE FILTER B (Third best dimension)
                if data[idx, dim3] >= q3_l and data[idx, dim3] <= q3_h:
                    
                    # FINAL STAGE: Full N-Dimensional Euclidean Distance
                    dist_sq = 0.0
                    for d in range(D):
                        diff = data[idx, d] - query_p[d]
                        dist_sq += diff * diff
                    
                    if dist_sq <= r_sq:
                        if count < max_c:
                            res_idx[count] = idx
                            res_dst[count] = np.sqrt(dist_sq)
                            count += 1

    if count == 0:
        return np.zeros(0, dtype=int64), np.zeros(0, dtype=float32)
    
    # Slice, sort and return K-nearest
    final_idx = res_idx[:count]
    final_dst = res_dst[:count]
    sort_idx = np.argsort(final_dst)
    return final_idx[sort_idx][:K], final_dst[sort_idx][:K]

class FastProximity:
    """
    Fast-Proximity Engine v2.0
    Smart adaptive neighbor search for dynamic high-dimensional datasets.
    """
    def __init__(self, data, grid_size=5000):
        self.data = data.astype(np.float32)
        self.N, self.D = data.shape
        self.grid_size = grid_size

        # 1. Heuristic: Select top 3 dimensions by variance
        variances = np.var(self.data, axis=0)
        self.dims = np.argsort(variances)[::-1][:3]

        # 2. Build 1D Grid on primary dimension (Self-Normalizing)
        d1 = self.dims[0]
        self.m1 = self.data[:, d1].min()
        max1 = self.data[:, d1].max()
        self.s1 = (max1 - self.m1) if max1 > self.m1 else 1.0
        
        # Calculate cell IDs
        c_ids1 = np.clip(((self.data[:, d1] - self.m1) / self.s1 * grid_size).astype(np.int32), 0, grid_size - 1)
        
        # Sort data pointers for fast grid access
        self.s_idx1 = np.argsort(c_ids1).astype(np.int64)
        ids1_sorted = c_ids1[self.s_idx1]
        
        # Find start/end pointers for each cell
        self.starts1 = np.searchsorted(ids1_sorted, np.arange(grid_size)).astype(np.intp)
        self.ends1 = np.append(self.starts1[1:], np.intp(self.N))

    def query(self, query_point, R, K=5):
        """
        Returns the K nearest neighbors within radius R using Triple-Filter logic.
        """
        # Prepare range boundaries for passive filters
        q2_l, q2_h = query_point[self.dims[1]] - R, query_point[self.dims[1]] + R
        q3_l, q3_h = query_point[self.dims[2]] - R, query_point[self.dims[2]] + R

        return _numba_triple_filter_query(
            self.data, 
            query_point.astype(np.float32), 
            float(R*R), 
            int(self.grid_size),
            self.starts1, 
            self.ends1, 
            self.s_idx1, 
            self.dims[0], 
            self.m1, 
            self.s1,
            self.dims[1], q2_l, q2_h,
            self.dims[2], q3_l, q3_h, 
            int(K)
        )
         
