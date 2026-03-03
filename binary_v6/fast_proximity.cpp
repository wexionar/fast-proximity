#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <random>
#include <numeric>

/**
 * FAST PROXIMITY v5.1 - "Recovery Edition"
 * Hardened version with extra safety checks.
 */

struct Neighbor {
    int64_t id;
    float dist_sq;
};

class FastProximity {
private:
    const float* data;
    size_t N, D;
    int grid_res;
    std::vector<int64_t> s_idx;
    std::vector<int> starts;
    std::vector<int> ends;
    int dims[3];
    float mins[2], scales[2];

public:
    FastProximity(const float* data_ptr, size_t n, size_t d, int res) 
        : data(data_ptr), N(n), D(d), grid_res(res) {
        
        // 1. AUTO-VARIANCE
        std::vector<double> sum_j(D, 0.0), sum_sq_j(D, 0.0);
        for(size_t i = 0; i < N; ++i) {
            for(size_t j = 0; j < D; ++j) {
                double val = data[i*D + j];
                sum_j[j] += val;
                sum_sq_j[j] += val * val;
            }
        }

        std::vector<std::pair<float, int>> var_idx;
        for(size_t j = 0; j < D; ++j) {
            float v = (float)((sum_sq_j[j] / N) - ((sum_j[j] / N) * (sum_j[j] / N)));
            var_idx.push_back({v, (int)j});
        }
        std::sort(var_idx.rbegin(), var_idx.rend());

        dims[0] = var_idx[0].second;
        dims[1] = (D > 1) ? var_idx[1].second : dims[0];
        dims[2] = (D > 2) ? var_idx[2].second : dims[1];

        // 2. STABLE NORMALIZATION
        float min1 = 1e30f, max1 = -1e30f, min2 = 1e30f, max2 = -1e30f;
        for(size_t i = 0; i < N; ++i) {
            min1 = std::min(min1, data[i*D + dims[0]]);
            max1 = std::max(max1, data[i*D + dims[0]]);
            min2 = std::min(min2, data[i*D + dims[1]]);
            max2 = std::max(max2, data[i*D + dims[1]]);
        }
        mins[0] = min1; 
        scales[0] = (max1 - min1) > 1e-9f ? (max1 - min1) : 1.0f;
        mins[1] = min2; 
        scales[1] = (max2 - min2) > 1e-9f ? (max2 - min2) : 1.0f;

        // 3. GRID CONSTRUCTION
        s_idx.resize(N);
        std::iota(s_idx.begin(), s_idx.end(), 0);
        std::vector<int> c_ids(N);
        for(size_t i = 0; i < N; ++i) {
            int cx = (int)std::floor((data[i*D + dims[0]] - mins[0]) / scales[0] * (grid_res - 1));
            int cy = (int)std::floor((data[i*D + dims[1]] - mins[1]) / scales[1] * (grid_res - 1));
            cx = std::clamp(cx, 0, grid_res - 1);
            cy = std::clamp(cy, 0, grid_res - 1);
            c_ids[i] = cx + cy * grid_res;
        }

        std::sort(s_idx.begin(), s_idx.end(), [&](int64_t a, int64_t b) {
            return c_ids[a] < c_ids[b];
        });

        starts.assign(grid_res * grid_res, -1);
        ends.assign(grid_res * grid_res, -1);
        for(size_t i = 0; i < N; ++i) {
            int cid = c_ids[s_idx[i]];
            if(starts[cid] == -1) starts[cid] = (int)i;
            ends[cid] = (int)i + 1;
        }
    }

    std::vector<Neighbor> query(const float* q, float R, int K) {
        float r_sq = R * R;
        std::vector<Neighbor> candidates;
        candidates.reserve(5000);

        int x0 = std::clamp((int)std::floor((q[dims[0]] - R - mins[0]) / scales[0] * (grid_res - 1)), 0, grid_res - 1);
        int x1 = std::clamp((int)std::floor((q[dims[0]] + R - mins[0]) / scales[0] * (grid_res - 1)), 0, grid_res - 1);
        int y0 = std::clamp((int)std::floor((q[dims[1]] - R - mins[1]) / scales[1] * (grid_res - 1)), 0, grid_res - 1);
        int y1 = std::clamp((int)std::floor((q[dims[1]] + R - mins[1]) / scales[1] * (grid_res - 1)), 0, grid_res - 1);

        for(int y = y0; y <= y1; ++y) {
            for(int x = x0; x <= x1; ++x) {
                int cid = x + y * grid_res;
                if(starts[cid] == -1) continue;

                for(int i = starts[cid]; i < ends[cid]; ++i) {
                    int64_t idx = s_idx[i];
                    const float* p = &data[idx * D];

                    if(D > 2 && std::abs(p[dims[2]] - q[dims[2]]) > R) continue;

                    float d_sq = 0;
                    bool skip = false;
                    for(size_t d = 0; d < D; ++d) {
                        float diff = p[d] - q[d];
                        d_sq += diff * diff;
                        if(d_sq > r_sq) { skip = true; break; }
                    }
                    if(!skip) candidates.push_back({idx, d_sq});
                }
            }
        }

        if(candidates.size() > (size_t)K) {
            std::partial_sort(candidates.begin(), candidates.begin() + K, candidates.end(), 
                [](const Neighbor& a, const Neighbor& b){ return a.dist_sq < b.dist_sq; });
            candidates.resize(K);
        } else {
            std::sort(candidates.begin(), candidates.end(), 
                [](const Neighbor& a, const Neighbor& b){ return a.dist_sq < b.dist_sq; });
        }
        return candidates;
    }
};

int main() {
    const size_t N = 1000000, D = 5;
    std::vector<float> data(N * D);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(0.0, 1.0);
    for(size_t i = 0; i < N * D; ++i) data[i] = dis(gen);

    FastProximity engine(data.data(), N, D, 200);

    std::vector<float> q(D);
    for(size_t d = 0; d < D; ++d) q[d] = dis(gen);
    float R = 0.2f;

    auto start = std::chrono::high_resolution_clock::now();
    auto results = engine.query(q.data(), R, 5);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> elapsed = end - start;
    std::cout << "--- FAST PROXIMITY v5.1 (CLAUDE RECOVERY) ---" << std::endl;
    std::cout << "Query time: " << elapsed.count() << " ms" << std::endl;
    std::cout << "Neighbors found: " << results.size() << std::endl;
    if(!results.empty()) {
        std::cout << "Closest ID: " << results[0].id << " | Dist: " << std::sqrt(results[0].dist_sq) << std::endl;
    }
    return 0;
}
