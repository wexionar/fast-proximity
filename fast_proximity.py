"""
================================================================================
FAST PROXIMITY v2.1 - "The Elastic Evolution"
High-performance, JIT-accelerated N-Dimensional spatial search engine.
Optimized for ultra-low latency neighbor queries and large-scale datasets.
================================================================================

SLRM Team: Alex · Gemini · ChatGPT · Claude · Grok · Meta AI
Version: 2.1 (Elastic Multi-Filter Engine)
License: MIT
================================================================================
"""

import numpy as np
from numba import njit, float32, int64, intp

@njit(fastmath=True)
def _numba_elastic_query(data, query_p, r_sq, gs, 
                          starts1, ends1, s_idx1, dim1, min1, scale1,
                          passive_dims, passive_ranges, K):
    """
    Elastic JIT-compiled search logic. 
    Adapts filtering stages based on available dimensions (1D to ND).
    """
    N, D = data.shape
    r_val = np.sqrt(r_sq)
    
    # 1. Active Grid Normalization (Primary Dim)
    q1_n = (query_p[dim1] - min1) / scale1
    r1_n = r_val / scale1
    
    x0 = int(max(0, (q1_n - r1_n) * gs))
    x1 = int(min(gs - 1, (q1_n + r1_n) * gs))
    
    max_c = 1000 
    res_idx = np.empty(max_c, dtype=int64)
    res_dst = np.empty(max_c, dtype=float32)
    count = 0

    num_passive = len(passive_dims)

    # 3. Execution Loop
    for x in range(x0, x1 + 1):
        for i in range(starts1[x], ends1[x]):
            idx = s_idx1[i]
            
            # ELASTIC PASSIVE FILTERS
            passed = True
            for p in range(num_passive):
                p_dim = passive_dims[p]
                if data[idx, p_dim] < passive_ranges[p, 0] or data[idx, p_dim] > passive_ranges[p, 1]:
                    passed = False
                    break
            
            if passed:
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
    
    final_idx = res_idx[:count]
    final_dst = res_dst[:count]
    sort_idx = np.argsort(final_dst)
    return final_idx[sort_idx][:K], final_dst[sort_idx][:K]

class FastProximity:
    def __init__(self, data, grid_size=5000):
        self.data = data.astype(np.float32)
        self.N, self.D = data.shape
        self.grid_size = grid_size

        # 1. Adaptive Heuristic: Select up to 3 dimensions by variance
        variances = np.var(self.data, axis=0)
        # Handle cases where D < 3
        n_dims_to_use = min(self.D, 3)
        self.dims = np.argsort(variances)[::-1][:n_dims_to_use]

        # 2. Build 1D Grid on primary dimension
        d1 = self.dims[0]
        self.m1 = self.data[:, d1].min()
        max1 = self.data[:, d1].max()
        self.s1 = (max1 - self.m1) if max1 > self.m1 else 1.0
        
        c_ids1 = np.clip(((self.data[:, d1] - self.m1) / self.s1 * grid_size).astype(np.int32), 0, grid_size - 1)
        self.s_idx1 = np.argsort(c_ids1).astype(np.int64)
        ids1_sorted = c_ids1[self.s_idx1]
        
        self.starts1 = np.searchsorted(ids1_sorted, np.arange(grid_size)).astype(np.intp)
        self.ends1 = np.append(self.starts1[1:], np.intp(self.N))

    def query(self, query_point, R, K=5):
        # Prepare passive dimensions and ranges dynamically
        passive_dims = self.dims[1:].astype(np.int64)
        num_p = len(passive_dims)
        passive_ranges = np.zeros((num_p, 2), dtype=np.float32)
        
        for i in range(num_p):
            d = passive_dims[i]
            passive_ranges[i, 0] = query_point[d] - R
            passive_ranges[i, 1] = query_point[d] + R

        return _numba_elastic_query(
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
            passive_dims,
            passive_ranges,
            int(K)
        )
      
