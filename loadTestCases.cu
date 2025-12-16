#include "include/mazeRouter.hpp"
//#include "include/util.hpp"
#include <iostream>

#include <algorithm>
#include <vector>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cstdlib>
#include <climits>
#include <cuda_runtime.h>

using namespace std;
using namespace std::chrono;

#include "include/mazeRouter.hpp"
#include "include/util.hpp"
#include <iostream>

#include <algorithm>
#include <vector>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cstdlib>
#include <climits>
#include <cuda_runtime.h>
#include <queue>

using namespace std;
using namespace std::chrono;

//const int INF = 1000000000;

// ==================== DATA STRUCTURES ====================
struct Point {
    int x, y;
    
    Point() : x(0), y(0) {}
    Point(int x_, int y_) : x(x_), y(y_) {}
    
    Point clamped(int N) const {
        return Point(max(0, min(N-1, x)), max(0, min(N-1, y)));
    }
    
    bool operator==(const Point& other) const {
        return x == other.x && y == other.y;
    }
    
    bool operator!=(const Point& other) const {
        return !(*this == other);
    }
};

struct PathResult {
    vector<Point> full_path;
    bool valid;
    int total_cost;
};

// ==================== DIAGONAL BRIDGING STRUCTURES ====================
struct TransformationInfo {
    Point normalized_source;
    Point normalized_sink;
    bool flip_x;      // Whether x-coordinate was flipped
    bool flip_y;      // Whether y-coordinate was flipped
    bool swapped;     // Whether source and sink were swapped
    int case_type;    // 1: TL->BR, 2: BR->TL, 3: TR->BL, 4: BL->TR, 5: colinear, 6: same point
    
    TransformationInfo() : flip_x(false), flip_y(false), swapped(false), case_type(0) {}
};

// Normalize points to TL->BR case for diagonal bridging
TransformationInfo normalizePoints(Point source, Point sink, int N) {
    TransformationInfo info;
    
    // Clamp points to grid boundaries
    source = source.clamped(N);
    sink = sink.clamped(N);
    
    // Check if points are the same
    if (source.x == sink.x && source.y == sink.y) {
        info.normalized_source = source;
        info.normalized_sink = sink;
        info.case_type = 6; // Same point
        return info;
    }
    
    // Check if points are colinear (horizontal or vertical)
    if (source.x == sink.x || source.y == sink.y) {
        info.normalized_source = source;
        info.normalized_sink = sink;
        info.case_type = 5; // Colinear
        return info;
    }
    
    // Determine relative position
    bool source_left = source.x < sink.x;
    bool source_top = source.y < sink.y;
    
    // Case 1: source is top-left, sink is bottom-right (no transformation needed)
    if (source_left && source_top) {
        info.normalized_source = source;
        info.normalized_sink = sink;
        info.case_type = 1;
        return info;
    }
    // Case 2: source is bottom-right, sink is top-left (swap them)
    else if (!source_left && !source_top) {
        info.normalized_source = sink;  // After swapping, original sink becomes source
        info.normalized_sink = source;  // Original source becomes sink
        info.swapped = true;
        info.case_type = 2;
        return info;
    }
    // Case 3: source is top-right, sink is bottom-left (flip horizontally)
    else if (!source_left && source_top) {
        // Flip horizontally: x' = N-1 - x
        info.normalized_source = Point(N-1 - source.x, source.y);
        info.normalized_sink = Point(N-1 - sink.x, sink.y);
        info.flip_x = true;
        info.case_type = 3;
        return info;
    }
    // Case 4: source is bottom-left, sink is top-right (flip vertically)
    else if (source_left && !source_top) {
        // Flip vertically: y' = N-1 - y
        info.normalized_source = Point(source.x, N-1 - source.y);
        info.normalized_sink = Point(sink.x, N-1 - sink.y);
        info.flip_y = true;
        info.case_type = 4;
        return info;
    }
    
    // Should never reach here
    info.normalized_source = source;
    info.normalized_sink = sink;
    info.case_type = 0;
    return info;
}

// Transform a point back to original coordinates
Point transformBack(Point p, const TransformationInfo& info, int N) {
    Point result = p;
    
    // Apply inverse transformations in reverse order
    if (info.flip_y) {
        result.y = N-1 - result.y;
    }
    if (info.flip_x) {
        result.x = N-1 - result.x;
    }
    
    return result;
}

// ==================== CPU DIJKSTRA FOR DIAGONAL BRIDGING ====================
class DijkstraSolverCPU {
private:
    struct HeapNode {
        int x, y;
        int dist;
        
        bool operator>(const HeapNode& other) const {
            return dist > other.dist;
        }
    };
    
    const int dx[4] = {1, -1, 0, 0};
    const int dy[4] = {0, 0, 1, -1};
    
public:
    int shortestPath(const vector<vector<int>>& grid, Point start, Point end, int N) {
        // Clamp points to grid
        start = start.clamped(N);
        end = end.clamped(N);
        
        // If start equals end, return the cost at that point
        if (start.x == end.x && start.y == end.y) {
            return grid[start.x][start.y];
        }
        
        vector<vector<int>> dist(N, vector<int>(N, INF));
        vector<vector<bool>> visited(N, vector<bool>(N, false));
        
        priority_queue<HeapNode, vector<HeapNode>, greater<HeapNode>> pq;
        
        dist[start.x][start.y] = grid[start.x][start.y];
        pq.push({start.x, start.y, dist[start.x][start.y]});
        
        while (!pq.empty()) {
            HeapNode current = pq.top();
            pq.pop();
            
            int x = current.x;
            int y = current.y;
            
            if (visited[x][y]) continue;
            visited[x][y] = true;
            
            if (x == end.x && y == end.y) {
                break;
            }
            
            for (int i = 0; i < 4; i++) {
                int nx = x + dx[i];
                int ny = y + dy[i];
                
                if (nx >= 0 && nx < N && ny >= 0 && ny < N) {
                    int new_dist = dist[x][y] + grid[nx][ny];
                    
                    if (new_dist < dist[nx][ny]) {
                        dist[nx][ny] = new_dist;
                        pq.push({nx, ny, new_dist});
                    }
                }
            }
        }
        
        return dist[end.x][end.y];
    }
    
    // Get actual path, not just cost
    vector<Point> getShortestPath(const vector<vector<int>>& grid, Point start, Point end, int N) {
        start = start.clamped(N);
        end = end.clamped(N);
        
        if (start.x == end.x && start.y == end.y) {
            return {start};
        }
        
        vector<vector<int>> dist(N, vector<int>(N, INF));
        vector<vector<Point>> prev(N, vector<Point>(N, Point(-1, -1)));
        vector<vector<bool>> visited(N, vector<bool>(N, false));
        
        priority_queue<HeapNode, vector<HeapNode>, greater<HeapNode>> pq;
        
        dist[start.x][start.y] = grid[start.x][start.y];
        pq.push({start.x, start.y, dist[start.x][start.y]});
        
        while (!pq.empty()) {
            HeapNode current = pq.top();
            pq.pop();
            
            int x = current.x;
            int y = current.y;
            
            if (visited[x][y]) continue;
            visited[x][y] = true;
            
            if (x == end.x && y == end.y) {
                break;
            }
            
            for (int i = 0; i < 4; i++) {
                int nx = x + dx[i];
                int ny = y + dy[i];
                
                if (nx >= 0 && nx < N && ny >= 0 && ny < N) {
                    int new_dist = dist[x][y] + grid[nx][ny];
                    
                    if (new_dist < dist[nx][ny]) {
                        dist[nx][ny] = new_dist;
                        prev[nx][ny] = Point(x, y);
                        pq.push({nx, ny, new_dist});
                    }
                }
            }
        }
        
        // Reconstruct path
        vector<Point> path;
        if (dist[end.x][end.y] < INF) {
            Point current = end;
            while (current.x != -1 && current.y != -1) {
                path.push_back(current);
                current = prev[current.x][current.y];
            }
            reverse(path.begin(), path.end());
        }
        
        return path;
    }
};

// ==================== GPU KERNEL WITH DIAGONAL BRIDGING ====================
__global__ void diagonalBridgingKernel(
    const int* cost_grid,
    int* path_exists,
    int* path_costs,
    int N,
    int num_pairs,
    int* sources_x,
    int* sources_y,
    int* dests_x,
    int* dests_y,
    int* diag_points_x,
    int* diag_points_y
) {
    int pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (pair_idx >= num_pairs) return;
    
    int start_x = sources_x[pair_idx];
    int start_y = sources_y[pair_idx];
    int end_x = dests_x[pair_idx];
    int end_y = dests_y[pair_idx];
    int diag_x = diag_points_x[pair_idx];
    int diag_y = diag_points_y[pair_idx];
    
    // Check bounds
    if (start_x < 0 || start_x >= N || start_y < 0 || start_y >= N ||
        end_x < 0 || end_x >= N || end_y < 0 || end_y >= N ||
        diag_x < 0 || diag_x >= N || diag_y < 0 || diag_y >= N) {
        path_exists[pair_idx] = 0;
        path_costs[pair_idx] = INF;
        return;
    }
    
    // If start equals end
    if (start_x == end_x && start_y == end_y) {
        path_exists[pair_idx] = 1;
        path_costs[pair_idx] = cost_grid[start_x * N + start_y];
        return;
    }
    
    // Check if start, end, or diagonal point is obstacle
    if (cost_grid[start_x * N + start_y] >= INF || 
        cost_grid[end_x * N + end_y] >= INF ||
        cost_grid[diag_x * N + diag_y] >= INF) {
        path_exists[pair_idx] = 0;
        path_costs[pair_idx] = INF;
        return;
    }
    
    // Check path from start to diagonal point
    bool path1_found = true;
    int current_x = start_x;
    int current_y = start_y;
    int indicates1 = -1; // 0 : HV, 1 : VH
    int indicates2 = -1; // 0 : HV, 1 : VH
    
    // Move horizontally first
    int step_x = (diag_x > current_x) ? 1 : -1;
    for (int x = current_x; x != diag_x; x += step_x) {
        if (cost_grid[x * N + current_y] >= INF) {
            path1_found = false;
            break;
        }
    }
    if (path1_found) {
        current_x = diag_x;
        int step_y = (diag_y > current_y) ? 1 : -1;
        for (int y = current_y; y != diag_y; y += step_y) {
            if (cost_grid[current_x * N + y] >= INF) {
                path1_found = false;
                break;
            }
        }
    }
    if (path1_found == true)
    {
        indicates1 = 0;
    }
    
    // If horizontal-vertical failed, try vertical-horizontal
    if (!path1_found) {
        path1_found = true;
        current_x = start_x;
        current_y = start_y;
        
        int step_y = (diag_y > current_y) ? 1 : -1;
        for (int y = current_y; y != diag_y; y += step_y) {
            if (cost_grid[current_x * N + y] >= INF) {
                path1_found = false;
                break;
            }
        }
        if (path1_found) {
            current_y = diag_y;
            int step_x = (diag_x > current_x) ? 1 : -1;
            for (int x = current_x; x != diag_x; x += step_x) {
                if (cost_grid[x * N + current_y] >= INF) {
                    path1_found = false;
                    break;
                }
            }
        }
        if (path1_found == true)
        {
            indicates1 = 1;
        }
    }
    
    // Check path from diagonal point to end
    bool path2_found = true;
    current_x = diag_x;
    current_y = diag_y;
    
    // Move horizontally first
    step_x = (end_x > current_x) ? 1 : -1;
    for (int x = current_x; x != end_x; x += step_x) {
        if (cost_grid[x * N + current_y] >= INF) {
            path2_found = false;
            break;
        }
    }
    if (path2_found) {
        current_x = end_x;
        int step_y = (end_y > current_y) ? 1 : -1;
        for (int y = current_y; y != end_y; y += step_y) {
            if (cost_grid[current_x * N + y] >= INF) {
                path2_found = false;
                break;
            }
        }
    }
    if(path2_found)
    {
        indicates2 = 0;
    }
    // If horizontal-vertical failed, try vertical-horizontal
    if (!path2_found) {
        path2_found = true;
        current_x = diag_x;
        current_y = diag_y;
        
        int step_y = (end_y > current_y) ? 1 : -1;
        for (int y = current_y; y != end_y; y += step_y) {
            if (cost_grid[current_x * N + y] >= INF) {
                path2_found = false;
                break;
            }
        }
        if (path2_found) {
            current_y = end_y;
            int step_x = (end_x > current_x) ? 1 : -1;
            for (int x = current_x; x != end_x; x += step_x) {
                if (cost_grid[x * N + current_y] >= INF) {
                    path2_found = false;
                    break;
                }
            }
        }
        if(path2_found)
        {
            indicates2 = 1;
        }
    }
    
    if (path1_found && path2_found) {
        path_exists[pair_idx] = 1;
        // Calculate total cost
        int cost = 0;
        
        // HV
        if(indicates1 == 0)
        {
            // Cost from start to diagonal
            current_x = start_x;
            current_y = start_y;
            step_x = (diag_x > current_x) ? 1 : -1;
            for (int x = current_x; x != diag_x; x += step_x) {
                cost += cost_grid[x * N + current_y];
            }
            current_x = diag_x;
            int step_y = (diag_y > current_y) ? 1 : -1;
            for (int y = current_y; y != diag_y; y += step_y) {
                cost += cost_grid[current_x * N + y];
            }
        }
        else if(indicates1 == 1)
        {
            // Cost from start to diagonal
            current_x = start_x;
            current_y = start_y;
            int step_y = (diag_y > current_y) ? 1 : -1;
            for (int y = current_y; y != diag_y; y += step_y) {
                cost += cost_grid[current_x * N + y];
            }
            current_y = diag_y;
            step_x = (diag_x > current_x) ? 1 : -1;
            for (int x = current_x; x != diag_x; x += step_x) {
                cost += cost_grid[x * N + current_y];
            }
        }
        // HV
        if(indicates2 == 0)
        {
            // Cost from diagonal to end
            current_x = diag_x;
            current_y = diag_y;
            step_x = (end_x > current_x) ? 1 : -1;
            for (int x = current_x; x != end_x; x += step_x) {
                cost += cost_grid[x * N + current_y];
            }
            current_x = end_x;
            int step_y = (end_y > current_y) ? 1 : -1;
            for (int y = current_y; y != end_y; y += step_y) {
                cost += cost_grid[current_x * N + y];
            }
        }
        // VH
        else if(indicates2 == 1)
        {
            // Cost from diagonal to end
            current_x = diag_x;
            current_y = diag_y;
            int step_y = (end_y > current_y) ? 1 : -1;
            for (int y = current_y; y != end_y; y += step_y) {
                cost += cost_grid[current_x * N + y];
            }
            current_y = end_y;
            step_x = (end_x > current_x) ? 1 : -1;
            for (int x = current_x; x != end_x; x += step_x) {
                cost += cost_grid[x * N + current_y];
            }
        }
        path_costs[pair_idx] = cost;
    } else {
        path_exists[pair_idx] = 0;
        path_costs[pair_idx] = INF;
    }
}

// ==================== GPU PATH FINDER WITH DIAGONAL BRIDGING ====================
class GPUPathFinderDiagonalBridging {
private:
    int* d_cost_grid;
    int* d_path_exists;
    int* d_path_costs;
    int* d_sources_x;
    int* d_sources_y;
    int* d_dests_x;
    int* d_dests_y;
    int* d_diag_points_x;
    int* d_diag_points_y;
    int grid_size;
    
    DijkstraSolverCPU cpu_dijkstra; // For fallback
    
public:
    GPUPathFinderDiagonalBridging(int N) : grid_size(N) {
        cudaMalloc(&d_cost_grid, sizeof(int) * N * N);
        cudaMalloc(&d_path_exists, sizeof(int) * 1000000);
        cudaMalloc(&d_path_costs, sizeof(int) * 1000000);
        cudaMalloc(&d_sources_x, sizeof(int) * 1000000);
        cudaMalloc(&d_sources_y, sizeof(int) * 1000000);
        cudaMalloc(&d_dests_x, sizeof(int) * 1000000);
        cudaMalloc(&d_dests_y, sizeof(int) * 1000000);
        cudaMalloc(&d_diag_points_x, sizeof(int) * 1000000);
        cudaMalloc(&d_diag_points_y, sizeof(int) * 1000000);
    }
    
    ~GPUPathFinderDiagonalBridging() {
        cudaFree(d_cost_grid);
        cudaFree(d_path_exists);
        cudaFree(d_path_costs);
        cudaFree(d_sources_x);
        cudaFree(d_sources_y);
        cudaFree(d_dests_x);
        cudaFree(d_dests_y);
        cudaFree(d_diag_points_x);
        cudaFree(d_diag_points_y);
    }
    
    vector<PathResult> findOptimalPathsGPUFast(
        const vector<pair<int, int>>& positions,
        const vector<vector<int>>& cost_grid,
        double& gpu_time_ms
    ) {
        auto start = high_resolution_clock::now();
        
        int num_points = positions.size();
        vector<PathResult> results(num_points - 1);
        
        if (num_points < 2) {
            gpu_time_ms = 0;
            return results;
        }
        
        // Flatten cost grid
        vector<int> flat_cost(grid_size * grid_size);
        for (int i = 0; i < grid_size; i++) {
            for (int j = 0; j < grid_size; j++) {
                flat_cost[i * grid_size + j] = cost_grid[i][j];
            }
        }
        
        // Copy cost grid to GPU
        cudaMemcpy(d_cost_grid, flat_cost.data(),
                  sizeof(int) * grid_size * grid_size, cudaMemcpyHostToDevice);
        
        // Prepare pairs for diagonal bridging
        vector<int> sources_x, sources_y, dests_x, dests_y;
        vector<int> diag_points_x, diag_points_y;
        vector<int> pair_indices;
        vector<TransformationInfo> trans_infos;
        
        for (int i = 0; i < num_points - 1; i++) {
            Point source = {positions[i].first, positions[i].second};
            Point sink = {positions[i+1].first, positions[i+1].second};
            
            // Get transformation info
            TransformationInfo trans_info = normalizePoints(source, sink, grid_size);
            
            // Handle colinear and same point cases with CPU Dijkstra
            if (trans_info.case_type == 5 || trans_info.case_type == 6) {
                // Use CPU Dijkstra for these cases
                vector<Point> path = cpu_dijkstra.getShortestPath(cost_grid, source, sink, grid_size);
                int cost = 0;
                for (const auto& p : path) {
                    cost += cost_grid[p.x][p.y];
                }
                
                results[i].full_path = path;
                results[i].valid = !path.empty();
                results[i].total_cost = cost;
                continue;
            }
            
            // For diagonal cases, try multiple diagonal points
            int dr = trans_info.normalized_sink.x - trans_info.normalized_source.x;
            int dc = trans_info.normalized_sink.y - trans_info.normalized_source.y;
            int n = max(dr, dc) + 1;
            
            // Try up to 5 diagonal points (to limit GPU workload)
            //int step = max(1, n / 5);
            // Use adaptive step:
            int step;
            if (n < 10) step = 1;        // Small square: test all
            else if (n < 20) step = 2;   // Medium: test half
            else step = n / 5;          // Large: test 10%
            for (int j = 0; j < n; j += step) {
                Point d_normalized = {
                    trans_info.normalized_source.x + j,
                    trans_info.normalized_source.y + (n - 1 - j) // Second diagonal
                };
                
                // Transform back to original coordinates
                Point d_original = transformBack(d_normalized, trans_info, grid_size);
                
                // Clamp to grid
                d_original = d_original.clamped(grid_size);
                
                sources_x.push_back(source.x);
                sources_y.push_back(source.y);
                dests_x.push_back(sink.x);
                dests_y.push_back(sink.y);
                diag_points_x.push_back(d_original.x);
                diag_points_y.push_back(d_original.y);
                pair_indices.push_back(i);
                trans_infos.push_back(trans_info);
            }
        }
        
        int num_pairs = sources_x.size();
        
        if (num_pairs > 0) {
            vector<int> h_path_exists(num_pairs, 0);
            vector<int> h_path_costs(num_pairs, INF);
            
            // Copy data to GPU
            cudaMemcpy(d_sources_x, sources_x.data(), sizeof(int) * num_pairs, cudaMemcpyHostToDevice);
            cudaMemcpy(d_sources_y, sources_y.data(), sizeof(int) * num_pairs, cudaMemcpyHostToDevice);
            cudaMemcpy(d_dests_x, dests_x.data(), sizeof(int) * num_pairs, cudaMemcpyHostToDevice);
            cudaMemcpy(d_dests_y, dests_y.data(), sizeof(int) * num_pairs, cudaMemcpyHostToDevice);
            cudaMemcpy(d_diag_points_x, diag_points_x.data(), sizeof(int) * num_pairs, cudaMemcpyHostToDevice);
            cudaMemcpy(d_diag_points_y, diag_points_y.data(), sizeof(int) * num_pairs, cudaMemcpyHostToDevice);
            
            // Launch kernel
            int block_size = 256;
            int grid_size_kernel = (num_pairs + block_size - 1) / block_size;
            
            diagonalBridgingKernel<<<grid_size_kernel, block_size>>>(
                d_cost_grid, d_path_exists, d_path_costs, grid_size, num_pairs,
                d_sources_x, d_sources_y, d_dests_x, d_dests_y,
                d_diag_points_x, d_diag_points_y
            );
            
            cudaDeviceSynchronize();
            
            // Check for errors
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                cerr << "CUDA error: " << cudaGetErrorString(err) << endl;
            } else {
                // Copy results back
                cudaMemcpy(h_path_exists.data(), d_path_exists, sizeof(int) * num_pairs, cudaMemcpyDeviceToHost);
                cudaMemcpy(h_path_costs.data(), d_path_costs, sizeof(int) * num_pairs, cudaMemcpyDeviceToHost);
                
                // Process results for each pair
                vector<vector<int>> candidate_costs(num_points - 1);
                vector<vector<Point>> candidate_points(num_points - 1);
                
                for (int p = 0; p < num_pairs; p++) {
                    int pair_idx = pair_indices[p];
                    if (h_path_exists[p] == 1) {
                        candidate_costs[pair_idx].push_back(h_path_costs[p]);
                        candidate_points[pair_idx].push_back(Point(diag_points_x[p], diag_points_y[p]));
                    }
                }
                
                // For each source-sink pair, choose best diagonal point
                for (int i = 0; i < num_points - 1; i++) {
                    if (results[i].valid) continue; // Already handled by CPU
                    
                    if (!candidate_costs[i].empty()) {
                        // Find minimum cost
                        int min_cost_idx = 0;
                        for (size_t j = 1; j < candidate_costs[i].size(); j++) {
                            if (candidate_costs[i][j] < candidate_costs[i][min_cost_idx]) {
                                min_cost_idx = j;
                            }
                        }
                        
                        Point best_diag = candidate_points[i][min_cost_idx];
                        Point source = {positions[i].first, positions[i].second};
                        Point sink = {positions[i+1].first, positions[i+1].second};
                        
                        // Create path: source -> diagonal -> sink
                        vector<Point> path = createLShapedPath(source, best_diag, cost_grid);
                        if (!path.empty()) {
                            // Remove the last point (diagonal) to avoid duplicate
                            path.pop_back();
                            
                            // Add path from diagonal to sink
                            vector<Point> path2 = createLShapedPath(best_diag, sink, cost_grid);
                            if (!path2.empty()) {
                                path.insert(path.end(), path2.begin(), path2.end());
                            }
                        }
                        
                        results[i].full_path = path;
                        results[i].valid = !path.empty();
                        results[i].total_cost = calculatePathCost(cost_grid, path);
                    } else {
                        // No valid diagonal path found, fallback to CPU Dijkstra
                        Point source = {positions[i].first, positions[i].second};
                        Point sink = {positions[i+1].first, positions[i+1].second};
                        
                        vector<Point> path = cpu_dijkstra.getShortestPath(cost_grid, source, sink, grid_size);
                        int cost = 0;
                        for (const auto& p : path) {
                            cost += cost_grid[p.x][p.y];
                        }
                        
                        results[i].full_path = path;
                        results[i].valid = !path.empty();
                        results[i].total_cost = cost;
                    }
                }
            }
        }
        
        // Handle any remaining pairs with CPU Dijkstra
        for (int i = 0; i < num_points - 1; i++) {
            if (!results[i].valid) {
                Point source = {positions[i].first, positions[i].second};
                Point sink = {positions[i+1].first, positions[i+1].second};
                
                vector<Point> path = cpu_dijkstra.getShortestPath(cost_grid, source, sink, grid_size);
                int cost = 0;
                for (const auto& p : path) {
                    cost += cost_grid[p.x][p.y];
                }
                
                results[i].full_path = path;
                results[i].valid = !path.empty();
                results[i].total_cost = cost;
            }
        }
        
        auto end = high_resolution_clock::now();
        gpu_time_ms = duration_cast<microseconds>(end - start).count() / 1000.0;
        
        return results;
    }
    
private:
    vector<Point> createLShapedPath(Point start, Point end, const vector<vector<int>>& grid) {
        vector<Point> path;
        
        start = start.clamped(grid_size);
        end = end.clamped(grid_size);
        
        if (grid[start.x][start.y] >= INF || grid[end.x][end.y] >= INF) {
            return {};
        }
        
        // Try horizontal then vertical
        vector<Point> path1 = tryHorizontalThenVertical(start, end, grid);
        vector<Point> path2 = tryVerticalThenHorizontal(start, end, grid);
        
        if (!path1.empty() && !path2.empty()) {
            return (path1.size() <= path2.size()) ? path1 : path2;
        } else if (!path1.empty()) {
            return path1;
        } else if (!path2.empty()) {
            return path2;
        }
        
        return {};
    }
    
    vector<Point> tryHorizontalThenVertical(Point start, Point end, const vector<vector<int>>& grid) {
        vector<Point> path;
        path.push_back(start);
        
        int current_x = start.x;
        int current_y = start.y;
        
        // Move horizontally
        while (current_x != end.x) {
            int next_x = (current_x < end.x) ? current_x + 1 : current_x - 1;
            if (grid[next_x][current_y] >= INF) return {};
            current_x = next_x;
            path.push_back(Point(current_x, current_y));
        }
        
        // Move vertically
        while (current_y != end.y) {
            int next_y = (current_y < end.y) ? current_y + 1 : current_y - 1;
            if (grid[current_x][next_y] >= INF) return {};
            current_y = next_y;
            path.push_back(Point(current_x, current_y));
        }
        
        return path;
    }
    
    vector<Point> tryVerticalThenHorizontal(Point start, Point end, const vector<vector<int>>& grid) {
        vector<Point> path;
        path.push_back(start);
        
        int current_x = start.x;
        int current_y = start.y;
        
        // Move vertically
        while (current_y != end.y) {
            int next_y = (current_y < end.y) ? current_y + 1 : current_y - 1;
            if (grid[current_x][next_y] >= INF) return {};
            current_y = next_y;
            path.push_back(Point(current_x, current_y));
        }
        
        // Move horizontally
        while (current_x != end.x) {
            int next_x = (current_x < end.x) ? current_x + 1 : current_x - 1;
            if (grid[next_x][current_y] >= INF) return {};
            current_x = next_x;
            path.push_back(Point(current_x, current_y));
        }
        
        return path;
    }
    
    int calculatePathCost(const vector<vector<int>>& grid, const vector<Point>& path) {
        if (path.empty()) return INF;
        
        int cost = 0;
        for (const auto& p : path) {
            if (grid[p.x][p.y] >= INF) return INF;
            cost += grid[p.x][p.y];
        }
        return cost;
    }
};

// ==================== MAZE ROUTER GPU WITH DIAGONAL BRIDGING ====================
class MazeRouterGPUWithDiagonal {
private:
    GPUPathFinderDiagonalBridging* gpu_finder;
    int current_grid_size;
    
public:
    MazeRouterGPUWithDiagonal() : gpu_finder(nullptr), current_grid_size(0) {}
    
    ~MazeRouterGPUWithDiagonal() {
        if (gpu_finder != nullptr) {
            delete gpu_finder;
        }
    }
    
    // Main routing interface with diagonal bridging
    double route(
        const vector<vector<int>>& cost,
        int N,
        const vector<pair<int, int>>& pins,
        vector<pair<int, int>>& gpures
    ) {
        // Initialize or reinitialize GPU finder
        if (gpu_finder == nullptr || current_grid_size != N) {
            if (gpu_finder != nullptr) {
                delete gpu_finder;
            }
            gpu_finder = new GPUPathFinderDiagonalBridging(N);
            current_grid_size = N;
        }
        
        // Run GPU path finding with diagonal bridging
        double gpu_time_ms = 0.0;
        auto path_results = gpu_finder->findOptimalPathsGPUFast(pins, cost, gpu_time_ms);
        
        // Convert results to required format
        gpures.clear();
        
        // Combine all paths into single result vector
        for (const auto& result : path_results) {
            if (result.valid) {
                for (const auto& point : result.full_path) {
                    gpures.push_back({point.x, point.y});
                }
            }
        }
        
        // If no paths found but we have pins, at least add the pin positions
        if (gpures.empty() && !pins.empty()) {
            for (const auto& pin : pins) {
                gpures.push_back(pin);
            }
        }
        
        // Remove duplicates
        std::sort(gpures.begin(), gpures.end());
        auto it = std::unique(gpures.begin(), gpures.end());
        gpures.erase(it, gpures.end());
        
        return gpu_time_ms;
    }
    
    // Check if GPU is available
    bool isAvailable() const {
        int device_count = 0;
        cudaError_t err = cudaGetDeviceCount(&device_count);
        return (err == cudaSuccess && device_count > 0);
    }
    
    // Get GPU device info
    string getGPUInfo() {
        if (!isAvailable()) return "No GPU available";
        
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        
        stringstream ss;
        ss << "GPU: " << prop.name << " (CC " << prop.major << "." << prop.minor << ")";
        ss << ", Memory: " << prop.totalGlobalMem / (1024*1024) << " MB";
        ss << ", Cores: " << prop.multiProcessorCount * _ConvertSMVer2Cores(prop.major, prop.minor);
        
        return ss.str();
    }
    
private:
    // Helper function for CUDA core calculation
    int _ConvertSMVer2Cores(int major, int minor) {
        struct SMtoCores {
            int sm; // 0xMm (hex), M = SM Major version, m = SM minor version
            int cores;
        };
        
        SMtoCores gpuArchCoresPerSM[] = {
            {0x30, 192}, {0x32, 192}, {0x35, 192}, {0x37, 192},
            {0x50, 128}, {0x52, 128}, {0x53, 128}, {0x60,  64},
            {0x61, 128}, {0x62, 128}, {0x70,  64}, {0x72,  64},
            {0x75,  64}, {0x80,  64}, {0x86, 128}, {0x87, 128},
            {0x89, 128}, {0x90, 128}, {-1, -1}
        };
        
        int index = 0;
        while (gpuArchCoresPerSM[index].sm != -1) {
            if (gpuArchCoresPerSM[index].sm == ((major << 4) + minor)) {
                return gpuArchCoresPerSM[index].cores;
            }
            index++;
        }
        return 128; // Default reasonable value
    }
};


int N, NumBlks, NumPins;
const int dx[] = {-1, 1, 0, 0};
const int dy[] = {0, 0, -1, 1};

void dfs(int x, int y, vector<vector<int>> &cost) {
    if(cost[x][y] != -1) return;
    cost[x][y] = 0;
    for(int d = 0; d < 4; d++) {
        int nx = x + dx[d], ny = y+ dy[d];
        if(0 <= nx && nx < N && 0 <= ny && ny < N)
            dfs(nx, ny, cost);
    }
}

int evaluate(const vector<pair<int, int>> &res, vector<vector<int>> cost, const int N) {   
    const int turnCost = 50; 
    int tot = 0;
    for(auto e : res) {
        assert(cost[e.first][e.second] != -1);
        tot += cost[e.first][e.second];
        cost[e.first][e.second] = -1;
    }
    int turnCount = 0;
    for(int i = 0; i < N; i++)
        for(int j = 0; j < N; j++) if(cost[i][j] == -1) {
            int test[2] = {0, 0};
            for(int d = 0; d < 4; d++) {
                int x = i + dx[d], y = j + dy[d];
                test[d / 2] |= (0 <= x && x < N && 0 <= y && y < N && cost[x][y] == -1);
            }
            turnCount += test[0] && test[1];
        }
    dfs(res[0].first, res[0].second, cost);
    tot += turnCount * turnCost;

    for(auto e : res) 
    {
        if(cost[e.first][e.second] == -1)
        {
            cout << "Encounter dangling (" << e.first << "," << e.second << ")" <<endl;
            tot = -1;
        }
    }
        
    return tot;
}

int main() {
    
    //cout << "Number of grid cells (N x N): " << endl;
    cin >> N;
    vector<vector<int>> cost(N, vector<int> (N));
    //cout << "Cost matrix (" << N << " x " << N << "): " << endl;
    for(int i = 0; i < N; i++)
        for(int j = 0; j < N; j++)
            cin >> cost[i][j];
    //cout << "Give the number of blocks that cannot be routed through: " << endl;
    cin >> NumBlks;
    for(int i = 0; i < NumBlks; i++) {
        int x1, y1, x2, y2;
        //cout << "Block " << i + 1 << " (x1 y1 x2 y2): " << endl;
        cin >> x1 >> y1 >> x2 >> y2;
        for(int a = x1; a <= x2; a++)
            for(int b = y1; b <= y2; b++)
                cost[a][b] = INF;
    }
    //cout << "Give the number of pins to be connected: " << endl;
    cin >> NumPins;
    vector<pair<int, int>> pins(NumPins), gpures, cpures;
    for(int i = 0; i < NumPins; i++)
    {
        //cout << "Pins (x y): " << endl;
        cin >> pins[i].first >> pins[i].second;
    }
    vector<vector<int>> costD(N, vector<int> (N));
    costD = cost;
    MazeRouter mazeRouter;
    auto cputime = mazeRouter.route(cost, N, pins, cpures);
    assert(checkAllPinsOnPath(cpures, pins));
    cout << "Original CPU version:\n" <<  "    time: " << cputime.first * 1.0 / CLOCKS_PER_SEC << "s\n    cost: " << evaluate(cpures, cost, N) << endl;
    MazeRouterGPUWithDiagonal mazeRouterGPU;
    
    // Check GPU availability
    if (!mazeRouterGPU.isAvailable()) {
        cout << "GPU not available, falling back to CPU" << endl;
        return 1;
    }
    
    cout << "GPU Info: " << mazeRouterGPU.getGPUInfo() << endl;
    
    // Route
    double gputime = mazeRouterGPU.route(costD, N, pins, gpures);
    
    // Print results
    cout << "GPU routing time: " << gputime << " ms" << endl;
    cout << "Result points: " << gpures.size() << endl;
    assert(checkAllPinsOnPath(gpures, pins));
    cout << "Current GPU version:\n" <<  "    time: " << gputime / 1000 << "s\n    cost: " << evaluate(gpures, cost, N) << endl;

    return 0;
}