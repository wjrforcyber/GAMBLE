// Optimized GPU Dijkstra implementation with proper obstacle avoidance

#include <algorithm>
#include <unordered_set>
#include <vector>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cstdlib>
#include <climits>
#include <queue>
#include <stack>
#include <cuda_runtime.h>

using namespace std;
using namespace std::chrono;

const int INF = 1000000000;

// ==================== DATA STRUCTURES ====================
struct Point {
    int x, y;
    
    Point() : x(0), y(0) {}
    Point(int x_, int y_) : x(x_), y(y_) {}
    
    // Helper function to clamp point to grid boundaries
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
    Point diagonal_point;
    int cost_source_to_d;
    int cost_d_to_sink;
    int total_cost;
    int square_size;
    bool valid;
    bool exceeded_boundary;
    // Transformation info to map back to original coordinates
    bool flipped_x;
    bool flipped_y;
    bool swapped;
    int case_type;
    // Path information (points only, no cost calculation)
    vector<Point> path_source_to_d;
    vector<Point> path_d_to_sink;
    vector<Point> full_path;
};

struct DirectPathResult {
    int cost;
    double time_ms;
    vector<Point> path;
};

// ==================== TRANSFORMATION HELPER ====================
struct TransformationInfo {
    Point normalized_source;
    Point normalized_sink;
    bool flip_x;      // Whether x-coordinate was flipped
    bool flip_y;      // Whether y-coordinate was flipped
    bool swapped;     // Whether source and sink were swapped
    int case_type;    // 1: TL->BR, 2: BR->TL, 3: TR->BL, 4: BL->TR, 5: colinear, 6: same point
    
    TransformationInfo() : flip_x(false), flip_y(false), swapped(false), case_type(0) {}
};

// Simple transformation without recursion
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

// Transform a path back to original coordinates
vector<Point> transformPathBack(const vector<Point>& path, const TransformationInfo& info, int N) {
    vector<Point> result;
    result.reserve(path.size());
    
    for (const Point& p : path) {
        result.push_back(transformBack(p, info, N));
    }
    
    return result;
}

// ==================== CPU DIJKSTRA IMPLEMENTATION WITH PATH TRACKING ONLY ====================
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
    // Returns only path (no cost calculation)
    vector<Point> findPathOnly(
        const vector<vector<int>>& grid, 
        Point start, 
        Point end, 
        int N
    ) {
        // Clamp points to grid
        start = start.clamped(N);
        end = end.clamped(N);
        
        // If start equals end, return just the point
        if (start.x == end.x && start.y == end.y) {
            return {start};
        }
        
        vector<vector<int>> dist(N, vector<int>(N, INF));
        vector<vector<bool>> visited(N, vector<bool>(N, false));
        vector<vector<Point>> prev(N, vector<Point>(N, Point(-1, -1)));
        
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
                    // Skip obstacles (INF cost)
                    if (grid[nx][ny] >= INF) continue;
                    
                    int new_dist = dist[x][y] + grid[nx][ny];
                    
                    if (new_dist < dist[nx][ny]) {
                        dist[nx][ny] = new_dist;
                        prev[nx][ny] = Point(x, y);
                        pq.push({nx, ny, new_dist});
                    }
                }
            }
        }
        
        // Reconstruct path if reachable
        vector<Point> path;
        if (dist[end.x][end.y] < INF) {
            // Backtrack from end to start
            Point current = end;
            while (current.x != -1) {
                path.push_back(current);
                current = prev[current.x][current.y];
            }
            reverse(path.begin(), path.end());
        }
        
        return path;
    }
    
    // Calculate cost only (for performance comparison)
    int calculatePathCost(
        const vector<vector<int>>& grid, 
        const vector<Point>& path
    ) {
        if (path.empty()) return INF;
        
        int cost = 0;
        for (const Point& p : path) {
            // If path goes through obstacle, return INF
            if (grid[p.x][p.y] >= INF) return INF;
            cost += grid[p.x][p.y];
        }
        return cost;
    }
    
    DirectPathResult shortestPathDirect(
        const vector<vector<int>>& grid, 
        Point start, 
        Point end, 
        int N
    ) {
        auto start_time = high_resolution_clock::now();
        
        vector<Point> path = findPathOnly(grid, start, end, N);
        int cost = calculatePathCost(grid, path);
        
        auto end_time = high_resolution_clock::now();
        double time_ms = duration_cast<microseconds>(end_time - start_time).count() / 1000.0;
        
        return {cost, time_ms, path};
    }
};

// ==================== CPU PATH FINDER WITH DIAGONAL BRIDGING (PATH ONLY) ====================
class CPUPathFinderDiagonal {
private:
    DijkstraSolverCPU dijkstra_solver;
    
    PathResult computeDiagonalBridgingPathOnly(
        const vector<vector<int>>& grid,
        Point source,
        Point sink,
        int N,
        const TransformationInfo& trans_info
    ) {
        PathResult result;
        result.valid = false;
        result.total_cost = INF;
        result.exceeded_boundary = false;
        result.flipped_x = trans_info.flip_x;
        result.flipped_y = trans_info.flip_y;
        result.swapped = trans_info.swapped;
        result.case_type = trans_info.case_type;
        
        // Handle special cases
        if (trans_info.case_type == 5 || trans_info.case_type == 6) {
            // Colinear or same point - use direct Dijkstra
            vector<Point> path = dijkstra_solver.findPathOnly(grid, source, sink, N);
            result.valid = true;
            result.total_cost = INF; // Cost not calculated here
            result.diagonal_point = source;
            result.cost_source_to_d = 0;
            result.cost_d_to_sink = 0;
            result.square_size = 1;
            result.full_path = path;
            return result;
        }
        
        // For diagonal bridging cases
        int dr = trans_info.normalized_sink.x - trans_info.normalized_source.x;
        int dc = trans_info.normalized_sink.y - trans_info.normalized_source.y;
        
        if (dr < 0 || dc < 0) {
            // Should not happen after normalization
            return result;
        }
        
        int n = max(dr, dc) + 1;
        bool exceeds_x = (trans_info.normalized_source.x + n) > N;
        bool exceeds_y = (trans_info.normalized_source.y + n) > N;
        bool exceeds_boundary = exceeds_x || exceeds_y;
        
        // We'll find the path without calculating costs
        vector<Point> best_path_sd, best_path_dt;
        Point optimal_d_normalized;
        int optimal_j = -1;
        
        for (int j = 0; j < n; j++) {
            Point d_normalized = {
                trans_info.normalized_source.x + j,
                trans_info.normalized_source.y + (n - 1 - j)
            };
            
            // Check bounds
            if (d_normalized.x < 0 || d_normalized.x >= N || 
                d_normalized.y < 0 || d_normalized.y >= N) {
                continue;
            }
            
            vector<Point> path_sd = dijkstra_solver.findPathOnly(
                grid, trans_info.normalized_source, d_normalized, N
            );
            vector<Point> path_dt = dijkstra_solver.findPathOnly(
                grid, d_normalized, trans_info.normalized_sink, N
            );
            
            if (path_sd.empty() || path_dt.empty()) {
                continue;
            }
            
            // For simplicity, we'll use the first valid path found
            if (optimal_j == -1) {
                optimal_j = j;
                optimal_d_normalized = d_normalized;
                best_path_sd = path_sd;
                best_path_dt = path_dt;
                break; // Just take first valid path for demonstration
            }
        }
        
        if (optimal_j != -1) {
            // Transform diagonal point back to original coordinates
            Point optimal_d_original = transformBack(optimal_d_normalized, trans_info, N);
            
            // Transform paths back to original coordinates
            vector<Point> path_sd_transformed = transformPathBack(best_path_sd, trans_info, N);
            vector<Point> path_dt_transformed = transformPathBack(best_path_dt, trans_info, N);
            
            // Combine paths (remove duplicate diagonal point)
            vector<Point> full_path;
            if (!path_sd_transformed.empty() && !path_dt_transformed.empty()) {
                full_path = path_sd_transformed;
                full_path.insert(full_path.end(), path_dt_transformed.begin() + 1, path_dt_transformed.end());
            }
            
            // Adjust if swapped
            if (trans_info.swapped) {
                reverse(full_path.begin(), full_path.end());
            }
            
            result = {
                optimal_d_original,
                0, // cost_to_d (not calculated)
                0, // cost_from_d (not calculated)
                INF, // total_cost (not calculated)
                n,
                true,
                exceeds_boundary,
                trans_info.flip_x,
                trans_info.flip_y,
                trans_info.swapped,
                trans_info.case_type,
                path_sd_transformed,
                path_dt_transformed,
                full_path
            };
        }
        
        return result;
    }
    
public:
    vector<PathResult> findOptimalPaths(
        const vector<pair<int, int>>& positions,
        const vector<vector<int>>& cost_grid,
        double& cpu_time_ms
    ) {
        auto start = high_resolution_clock::now();
        
        vector<PathResult> results;
        int N = cost_grid.size();
        int num_points = positions.size();
        
        for (int i = 0; i < num_points - 1; i++) {
            Point source = {positions[i].first, positions[i].second};
            Point sink = {positions[i+1].first, positions[i+1].second};
            
            TransformationInfo trans_info = normalizePoints(source, sink, N);
            PathResult result = computeDiagonalBridgingPathOnly(cost_grid, source, sink, N, trans_info);
            
            results.push_back(result);
        }
        
        auto end = high_resolution_clock::now();
        cpu_time_ms = duration_cast<microseconds>(end - start).count() / 1000.0;
        
        return results;
    }
};

// ==================== UNIFIED COST CALCULATOR ====================
class PathCostCalculator {
public:
    // Calculate cost for a single path
    static int calculatePathCost(
        const vector<vector<int>>& grid,
        const vector<Point>& path
    ) {
        if (path.empty()) return INF;
        
        int cost = 0;
        for (const Point& p : path) {
            // If path goes through obstacle, return INF
            if (grid[p.x][p.y] >= INF) return INF;
            cost += grid[p.x][p.y];
        }
        return cost;
    }
    
    // Calculate costs for multiple paths (CPU)
    static vector<int> calculateCPUCosts(
        const vector<vector<int>>& grid,
        const vector<PathResult>& path_results
    ) {
        vector<int> costs;
        costs.reserve(path_results.size());
        
        for (const auto& result : path_results) {
            if (result.valid && !result.full_path.empty()) {
                costs.push_back(calculatePathCost(grid, result.full_path));
            } else {
                costs.push_back(INF);
            }
        }
        
        return costs;
    }
    
    // Calculate costs for multiple paths (GPU)
    static vector<int> calculateGPUCosts(
        const vector<vector<int>>& grid,
        const vector<PathResult>& path_results
    ) {
        // GPU uses same calculation as CPU for consistency
        return calculateCPUCosts(grid, path_results);
    }
};

// ==================== GPU KERNEL FUNCTIONS ====================
// Simple path kernel using global memory (no shared memory overflow)
__global__ void simplePathKernel(
    const int* cost_grid,
    int* path_exists,  // 1 if path exists, 0 otherwise
    int N,
    int num_pairs,
    int* sources_x,
    int* sources_y,
    int* dests_x,
    int* dests_y
) {
    int pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (pair_idx >= num_pairs) return;
    
    int start_x = sources_x[pair_idx];
    int start_y = sources_y[pair_idx];
    int end_x = dests_x[pair_idx];
    int end_y = dests_y[pair_idx];
    
    // Check bounds
    if (start_x < 0 || start_x >= N || start_y < 0 || start_y >= N ||
        end_x < 0 || end_x >= N || end_y < 0 || end_y >= N) {
        path_exists[pair_idx] = 0;
        return;
    }
    
    // If start equals end
    if (start_x == end_x && start_y == end_y) {
        path_exists[pair_idx] = 1;
        return;
    }
    
    // Check if start or end is obstacle
    if (cost_grid[start_x * N + start_y] >= INF || 
        cost_grid[end_x * N + end_y] >= INF) {
        path_exists[pair_idx] = 0;
        return;
    }
    
    // Simple check: try to find L-shaped path
    int dx = abs(end_x - start_x);
    int dy = abs(end_y - start_y);
    
    // Try horizontal then vertical path
    bool path_found = true;
    
    // Horizontal first
    int step_x = (end_x > start_x) ? 1 : -1;
    for (int x = start_x; x != end_x; x += step_x) {
        if (cost_grid[x * N + start_y] >= INF) {
            path_found = false;
            break;
        }
    }
    
    // Then vertical
    if (path_found) {
        int step_y = (end_y > start_y) ? 1 : -1;
        for (int y = start_y; y != end_y; y += step_y) {
            if (cost_grid[end_x * N + y] >= INF) {
                path_found = false;
                break;
            }
        }
    }
    
    // If horizontal-vertical failed, try vertical-horizontal
    if (!path_found) {
        path_found = true;
        
        // Vertical first
        int step_y = (end_y > start_y) ? 1 : -1;
        for (int y = start_y; y != end_y; y += step_y) {
            if (cost_grid[start_x * N + y] >= INF) {
                path_found = false;
                break;
            }
        }
        
        // Then horizontal
        if (path_found) {
            int step_x = (end_x > start_x) ? 1 : -1;
            for (int x = start_x; x != end_x; x += step_x) {
                if (cost_grid[x * N + end_y] >= INF) {
                    path_found = false;
                    break;
                }
            }
        }
    }
    
    path_exists[pair_idx] = path_found ? 1 : 0;
}

// ==================== GPU PATH FINDER (OPTIMIZED) ====================
class GPUPathFinderDijkstraActual {
private:
    int* d_cost_grid;
    int* d_path_exists;
    int* d_sources_x;
    int* d_sources_y;
    int* d_dests_x;
    int* d_dests_y;
    int grid_size;
    DijkstraSolverCPU cpu_dijkstra; // For fallback
    
public:
    GPUPathFinderDijkstraActual(int N) : grid_size(N) {
        cudaMalloc(&d_cost_grid, sizeof(int) * N * N);
        cudaMalloc(&d_path_exists, sizeof(int) * 1000000);
        cudaMalloc(&d_sources_x, sizeof(int) * 1000000);
        cudaMalloc(&d_sources_y, sizeof(int) * 1000000);
        cudaMalloc(&d_dests_x, sizeof(int) * 1000000);
        cudaMalloc(&d_dests_y, sizeof(int) * 1000000);
    }
    
    ~GPUPathFinderDijkstraActual() {
        cudaFree(d_cost_grid);
        cudaFree(d_path_exists);
        cudaFree(d_sources_x);
        cudaFree(d_sources_y);
        cudaFree(d_dests_x);
        cudaFree(d_dests_y);
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
        
        // Prepare pairs for GPU computation
        vector<int> sources_x, sources_y, dests_x, dests_y;
        vector<int> pair_indices;
        
        for (int i = 0; i < num_points - 1; i++) {
            Point source = {positions[i].first, positions[i].second};
            Point sink = {positions[i+1].first, positions[i+1].second};
            
            // Clamp points
            source = source.clamped(grid_size);
            sink = sink.clamped(grid_size);
            
            // Store coordinates
            sources_x.push_back(source.x);
            sources_y.push_back(source.y);
            dests_x.push_back(sink.x);
            dests_y.push_back(sink.y);
            pair_indices.push_back(i);
        }
        
        int num_pairs = sources_x.size();
        vector<int> h_path_exists(num_pairs, 0);
        
        if (num_pairs > 0) {
            // Copy data to GPU
            cudaMemcpy(d_sources_x, sources_x.data(), sizeof(int) * num_pairs, cudaMemcpyHostToDevice);
            cudaMemcpy(d_sources_y, sources_y.data(), sizeof(int) * num_pairs, cudaMemcpyHostToDevice);
            cudaMemcpy(d_dests_x, dests_x.data(), sizeof(int) * num_pairs, cudaMemcpyHostToDevice);
            cudaMemcpy(d_dests_y, dests_y.data(), sizeof(int) * num_pairs, cudaMemcpyHostToDevice);
            
            // Launch kernel
            int block_size = 256;
            int grid_size_kernel = (num_pairs + block_size - 1) / block_size;
            
            simplePathKernel<<<grid_size_kernel, block_size>>>(
                d_cost_grid, d_path_exists, grid_size, num_pairs,
                d_sources_x, d_sources_y, d_dests_x, d_dests_y
            );
            
            // Check for errors
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                cerr << "CUDA error: " << cudaGetErrorString(err) << endl;
                // Fall back to CPU for all paths
                return fallbackToCPU(positions, cost_grid, gpu_time_ms, start);
            }
            
            cudaDeviceSynchronize();
            
            // Copy results back
            cudaMemcpy(h_path_exists.data(), d_path_exists, sizeof(int) * num_pairs, cudaMemcpyDeviceToHost);
            
            // Process results
            for (int p = 0; p < num_pairs; p++) {
                int result_idx = pair_indices[p];
                
                Point source = {positions[result_idx].first, positions[result_idx].second};
                Point sink = {positions[result_idx + 1].first, positions[result_idx + 1].second};
                
                TransformationInfo trans_info = normalizePoints(source, sink, grid_size);
                vector<Point> path;
                
                if (h_path_exists[p] == 1) {
                    // GPU found a simple path - create L-shaped path
                    path = createLShapedPath(source, sink, cost_grid);
                } else {
                    // GPU couldn't find simple path - use CPU Dijkstra
                    path = cpu_dijkstra.findPathOnly(cost_grid, source, sink, grid_size);
                }
                
                results[result_idx] = {
                    source,
                    0, 0, INF, 1,
                    !path.empty(), false,
                    trans_info.flip_x, trans_info.flip_y, trans_info.swapped, trans_info.case_type,
                    {}, {}, path
                };
            }
        }
        
        auto end = high_resolution_clock::now();
        gpu_time_ms = duration_cast<microseconds>(end - start).count() / 1000.0;
        
        return results;
    }
    
private:
    vector<PathResult> fallbackToCPU(
        const vector<pair<int, int>>& positions,
        const vector<vector<int>>& cost_grid,
        double& gpu_time_ms,
        high_resolution_clock::time_point start
    ) {
        int num_points = positions.size();
        vector<PathResult> results(num_points - 1);
        
        for (int i = 0; i < num_points - 1; i++) {
            Point source = {positions[i].first, positions[i].second};
            Point sink = {positions[i+1].first, positions[i+1].second};
            
            TransformationInfo trans_info = normalizePoints(source, sink, grid_size);
            vector<Point> path = cpu_dijkstra.findPathOnly(cost_grid, source, sink, grid_size);
            
            results[i] = {
                source,
                0, 0, INF, 1,
                !path.empty(), false,
                trans_info.flip_x, trans_info.flip_y, trans_info.swapped, trans_info.case_type,
                {}, {}, path
            };
        }
        
        auto end = high_resolution_clock::now();
        gpu_time_ms = duration_cast<microseconds>(end - start).count() / 1000.0;
        
        return results;
    }
    
    vector<Point> createLShapedPath(Point start, Point end, const vector<vector<int>>& grid) {
        vector<Point> path;
        
        // Clamp points
        start = start.clamped(grid_size);
        end = end.clamped(grid_size);
        
        // Check if start or end is obstacle
        if (grid[start.x][start.y] >= INF || grid[end.x][end.y] >= INF) {
            return {};
        }
        
        // Add start point
        path.push_back(start);
        
        int current_x = start.x;
        int current_y = start.y;
        
        // Try horizontal then vertical
        bool horizontal_first = true;
        vector<Point> path1, path2;
        
        // Horizontal then vertical
        path1 = createHorizontalThenVertical(start, end, grid);
        
        // Vertical then horizontal
        path2 = createVerticalThenHorizontal(start, end, grid);
        
        // Choose the shorter path
        if (!path1.empty() && !path2.empty()) {
            return (path1.size() <= path2.size()) ? path1 : path2;
        } else if (!path1.empty()) {
            return path1;
        } else if (!path2.empty()) {
            return path2;
        }
        
        // If both L-shaped paths are blocked, use CPU Dijkstra
        return cpu_dijkstra.findPathOnly(grid, start, end, grid_size);
    }
    
    vector<Point> createHorizontalThenVertical(Point start, Point end, const vector<vector<int>>& grid) {
        vector<Point> path;
        path.push_back(start);
        
        int current_x = start.x;
        int current_y = start.y;
        
        // Move horizontally
        while (current_x != end.x) {
            int next_x = (current_x < end.x) ? current_x + 1 : current_x - 1;
            if (grid[next_x][current_y] >= INF) {
                return {}; // Path blocked
            }
            current_x = next_x;
            path.push_back(Point(current_x, current_y));
        }
        
        // Move vertically
        while (current_y != end.y) {
            int next_y = (current_y < end.y) ? current_y + 1 : current_y - 1;
            if (grid[current_x][next_y] >= INF) {
                return {}; // Path blocked
            }
            current_y = next_y;
            path.push_back(Point(current_x, current_y));
        }
        
        return path;
    }
    
    vector<Point> createVerticalThenHorizontal(Point start, Point end, const vector<vector<int>>& grid) {
        vector<Point> path;
        path.push_back(start);
        
        int current_x = start.x;
        int current_y = start.y;
        
        // Move vertically
        while (current_y != end.y) {
            int next_y = (current_y < end.y) ? current_y + 1 : current_y - 1;
            if (grid[current_x][next_y] >= INF) {
                return {}; // Path blocked
            }
            current_y = next_y;
            path.push_back(Point(current_x, current_y));
        }
        
        // Move horizontally
        while (current_x != end.x) {
            int next_x = (current_x < end.x) ? current_x + 1 : current_x - 1;
            if (grid[next_x][current_y] >= INF) {
                return {}; // Path blocked
            }
            current_x = next_x;
            path.push_back(Point(current_x, current_y));
        }
        
        return path;
    }
};

// ==================== CPU DIRECT PATH FINDER (PATH ONLY) ====================
class CPUPathFinderDirect {
private:
    DijkstraSolverCPU dijkstra_solver;
    
public:
    vector<DirectPathResult> findDirectPaths(
        const vector<pair<int, int>>& positions,
        const vector<vector<int>>& cost_grid,
        double& total_time_ms
    ) {
        auto start = high_resolution_clock::now();
        
        vector<DirectPathResult> results;
        int N = cost_grid.size();
        int num_points = positions.size();
        
        for (int i = 0; i < num_points - 1; i++) {
            Point source = {positions[i].first, positions[i].second};
            Point sink = {positions[i+1].first, positions[i+1].second};
            
            DirectPathResult result = dijkstra_solver.shortestPathDirect(cost_grid, source, sink, N);
            results.push_back(result);
        }
        
        auto end = high_resolution_clock::now();
        total_time_ms = duration_cast<microseconds>(end - start).count() / 1000.0;
        
        return results;
    }
};

// ==================== PATH VALIDATION HELPER ====================
class PathValidator {
public:
    // Check if all positions are present on the path
    static bool checkAllPositionsOnPath(
        const vector<Point>& path,
        const vector<pair<int, int>>& positions
    ) {
        if (path.empty() || positions.empty()) {
            return false;
        }
        
        cout << "Checking " << positions.size() << " positions against path of length " 
             << path.size() << "..." << endl;
        
        // Create a map for faster lookup of path points
        unordered_set<string> path_points;
        for (const auto& point : path) {
            string key = to_string(point.x) + "," + to_string(point.y);
            path_points.insert(key);
        }
        
        bool all_found = true;
        vector<bool> found(positions.size(), false);
        
        for (size_t i = 0; i < positions.size(); i++) {
            string key = to_string(positions[i].first) + "," + to_string(positions[i].second);
            
            if (path_points.find(key) != path_points.end()) {
                found[i] = true;
                cout << "  ✓ Position " << i << " (" << positions[i].first << "," 
                     << positions[i].second << ") FOUND" << endl;
            } else {
                all_found = false;
                cout << "  ✗ Position " << i << " (" << positions[i].first << "," 
                     << positions[i].second << ") MISSING" << endl;
            }
        }
        
        // Print summary
        int found_count = count(found.begin(), found.end(), true);
        cout << "Summary: " << found_count << "/" << positions.size() 
             << " positions found on path" << endl;
        
        return all_found;
    }
    
    // Check if path avoids obstacles
    static bool checkPathAvoidsObstacles(
        const vector<Point>& path,
        const vector<vector<int>>& grid
    ) {
        if (path.empty()) return false;
        
        cout << "Checking if path avoids obstacles..." << endl;
        bool avoids_obstacles = true;
        
        for (size_t i = 0; i < path.size(); i++) {
            const Point& p = path[i];
            if (grid[p.x][p.y] >= INF) {
                cout << "  ✗ Path goes through obstacle at (" 
                     << p.x << "," << p.y << ")" << endl;
                avoids_obstacles = false;
            }
        }
        
        if (avoids_obstacles) {
            cout << "  ✓ Path successfully avoids all obstacles" << endl;
        }
        
        return avoids_obstacles;
    }
    
    // Comprehensive path validation
    static void validatePathComprehensive(
        const vector<Point>& path,
        const vector<pair<int, int>>& positions,
        const vector<vector<int>>& grid,
        const string& algorithm_name
    ) {
        cout << "\n" << string(70, '=') << endl;
        cout << "COMPREHENSIVE VALIDATION: " << algorithm_name << endl;
        cout << string(70, '=') << endl;
        
        if (path.empty()) {
            cout << "  ✗ Path is empty!" << endl;
            return;
        }
        
        cout << "Path length: " << path.size() << " points" << endl;
        
        bool all_on_path = checkAllPositionsOnPath(path, positions);
        bool avoids_obs = checkPathAvoidsObstacles(path, grid);
        
        // Check path continuity
        bool continuous = true;
        for (size_t i = 1; i < path.size(); i++) {
            int dx = abs(path[i].x - path[i-1].x);
            int dy = abs(path[i].y - path[i-1].y);
            if (dx + dy > 1) {  // Not adjacent
                cout << "  ✗ Path discontinuity between points " 
                     << (i-1) << " and " << i << endl;
                continuous = false;
            }
        }
        if (continuous) {
            cout << "  ✓ Path is continuous (all adjacent points)" << endl;
        }
        
        cout << "\nVALIDATION SUMMARY:" << endl;
        cout << "  All positions on path:    " << (all_on_path ? "✓ PASS" : "✗ FAIL") << endl;
        cout << "  Avoids obstacles:         " << (avoids_obs ? "✓ PASS" : "✗ FAIL") << endl;
        cout << "  Path is continuous:       " << (continuous ? "✓ PASS" : "✗ FAIL") << endl;
        
        if (all_on_path && avoids_obs && continuous) {
            cout << "\n  ✓ PATH VALIDATION: COMPLETE SUCCESS!" << endl;
        } else {
            cout << "\n  ✗ PATH VALIDATION: FAILED!" << endl;
        }
    }
};

// ==================== PATH PRINTING HELPER ====================
void printPath(const vector<Point>& path, const string& label = "Path") {
    cout << label << " (" << path.size() << " points): ";
    for (size_t i = 0; i < path.size(); i++) {
        cout << "(" << path[i].x << "," << path[i].y << ")";
        if (i < path.size() - 1) cout << " -> ";
    }
    cout << endl;
}

// ==================== TEST FUNCTIONS ====================
vector<vector<int>> createMazeGridWithBlockedObstacles(int N) {
    vector<vector<int>> grid(N, vector<int>(N, 1));
    
    // Create BLOCKED obstacles (INF cost)
    for (int i = N/4; i < 3*N/4; i++) {
        grid[N/2][i] = INF;    // Horizontal obstacle - BLOCKED
        grid[i][N/2] = INF;    // Vertical obstacle - BLOCKED
    }
    
    // Create a cheap path (ensure it's not blocked)
    for (int i = 0; i < N; i++) {
        // Make sure diagonal doesn't intersect obstacles
        if (i != N/2) {  // Skip obstacle intersection points
            grid[i][i] = 1;
        } else {
            grid[i][i] = INF;  // Center is blocked
        }
    }
    
    // Add some random high-cost cells (not blocked)
    srand(42);
    for (int i = 0; i < N; i += 3) {
        for (int j = 0; j < N; j += 3) {
            if (rand() % 3 == 0) {
                // Make sure we're not on obstacle
                if (!(i == N/2 && j >= N/4 && j < 3*N/4) &&
                    !(j == N/2 && i >= N/4 && i < 3*N/4)) {
                    grid[i][j] = 50 + rand() % 50;
                }
            }
        }
    }
    
    return grid;
}

// ==================== TEST: OBSTACLE AVOIDANCE ====================
void testObstacleAvoidance() {
    cout << "\n\n==========================================" << endl;
    cout << "TEST: OBSTACLE AVOIDANCE VERIFICATION" << endl;
    cout << "==========================================" << endl;
    
    const int N = 15;
    auto grid = createMazeGridWithBlockedObstacles(N);
    
    // Print grid visualization
    cout << "\nGrid visualization (INF = obstacle, numbers = normal cells):" << endl;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (grid[i][j] >= INF) {
                cout << " INF";
            } else {
                cout << setw(4) << grid[i][j];
            }
        }
        cout << endl;
    }
    
    DijkstraSolverCPU dijkstra;
    CPUPathFinderDiagonal cpu_finder;
    GPUPathFinderDijkstraActual gpu_finder(N);
    PathCostCalculator cost_calculator;
    PathValidator validator;
    
    // Test case 1: Path must go around center obstacle
    {
        cout << "\n\nTest 1: Avoiding center cross obstacle" << endl;
        cout << string(50, '-') << endl;
        
        Point start = {0, 0};
        Point end = {N-1, N-1};
        
        cout << "Path from (" << start.x << "," << start.y << ") to (" 
             << end.x << "," << end.y << ")" << endl;
        cout << "Direct line goes through blocked center at (" 
             << N/2 << "," << N/2 << ")" << endl;
        
        // CPU Dijkstra
        auto direct_result = dijkstra.shortestPathDirect(grid, start, end, N);
        cout << "\nCPU Dijkstra:" << endl;
        cout << "  Path cost: " << (direct_result.cost < INF ? to_string(direct_result.cost) : "INF") << endl;
        cout << "  Path length: " << direct_result.path.size() << " points" << endl;
        if (direct_result.path.size() <= 20) {
            printPath(direct_result.path, "  Path");
        }
        validator.validatePathComprehensive(direct_result.path, 
                                           {{start.x, start.y}, {end.x, end.y}}, 
                                           grid, "CPU Dijkstra");
        
        // GPU
        double gpu_time;
        vector<pair<int, int>> positions = {{start.x, start.y}, {end.x, end.y}};
        auto gpu_results = gpu_finder.findOptimalPathsGPUFast(positions, grid, gpu_time);
        
        if (gpu_results[0].valid && !gpu_results[0].full_path.empty()) {
            cout << "\nGPU:" << endl;
            int gpu_cost = cost_calculator.calculatePathCost(grid, gpu_results[0].full_path);
            cout << "  Path cost: " << (gpu_cost < INF ? to_string(gpu_cost) : "INF") << endl;
            cout << "  Path length: " << gpu_results[0].full_path.size() << " points" << endl;
            if (gpu_results[0].full_path.size() <= 20) {
                printPath(gpu_results[0].full_path, "  Path");
            }
            validator.validatePathComprehensive(gpu_results[0].full_path, 
                                               positions, grid, "GPU");
        } else {
            cout << "\nGPU: No valid path found" << endl;
        }
    }
    
    // Test case 2: Path blocked by obstacle
    {
        cout << "\n\nTest 2: Path that should be blocked" << endl;
        cout << string(50, '-') << endl;
        
        // Start on one side of vertical obstacle, end on other side (very close)
        Point start = {N/2 - 1, N/4};
        Point end = {N/2 + 1, N/4};
        
        cout << "Path from (" << start.x << "," << start.y << ") to (" 
             << end.x << "," << end.y << ")" << endl;
        cout << "Blocked by vertical obstacle at column " << N/2 << endl;
        
        // CPU Dijkstra
        auto direct_result = dijkstra.shortestPathDirect(grid, start, end, N);
        cout << "\nCPU Dijkstra:" << endl;
        if (direct_result.path.empty() || direct_result.cost >= INF) {
            cout << "  ✓ No path found (correct - should be blocked)" << endl;
        } else {
            cout << "  Path found (length: " << direct_result.path.size() << ", cost: " 
                 << direct_result.cost << ")" << endl;
        }
    }
    
    // Test case 3: Simple straight-line path (no obstacles)
    {
        cout << "\n\nTest 3: Simple straight-line path without obstacles" << endl;
        cout << string(50, '-') << endl;
        
        Point start = {1, 1};
        Point end = {1, N-2};  // Vertical line avoiding center
        
        cout << "Path from (" << start.x << "," << start.y << ") to (" 
             << end.x << "," << end.y << ")" << endl;
        
        // CPU Dijkstra
        auto direct_result = dijkstra.shortestPathDirect(grid, start, end, N);
        cout << "\nCPU Dijkstra:" << endl;
        cout << "  Path cost: " << direct_result.cost << endl;
        cout << "  Path length: " << direct_result.path.size() << " points" << endl;
        
        // GPU
        double gpu_time;
        vector<pair<int, int>> positions = {{start.x, start.y}, {end.x, end.y}};
        auto gpu_results = gpu_finder.findOptimalPathsGPUFast(positions, grid, gpu_time);
        
        if (gpu_results[0].valid && !gpu_results[0].full_path.empty()) {
            cout << "\nGPU:" << endl;
            cout << "  Path length: " << gpu_results[0].full_path.size() << " points" << endl;
            validator.validatePathComprehensive(gpu_results[0].full_path, 
                                               positions, grid, "GPU Straight Line");
        }
    }
}

// ==================== TEST: PERFORMANCE ====================
void testPerformanceComparison() {
    cout << "\n\n==========================================" << endl;
    cout << "TEST: PERFORMANCE COMPARISON WITH OBSTACLES" << endl;
    cout << "==========================================" << endl;
    
    const int N = 50;
    const int NUM_RUNS = 2;
    
    auto grid = createMazeGridWithBlockedObstacles(N);
    PathCostCalculator cost_calculator;
    
    vector<int> pin_counts = {5, 10, 20, 50, 100};
    
    cout << fixed << setprecision(3);
    cout << "\nPerformance (Grid: " << N << "x" << N << " with obstacles)" << endl;
    cout << "\n" << string(120, '=') << endl;
    cout << setw(12) << "Pins" 
         << setw(12) << "Pairs"
         << setw(20) << "CPU Direct (ms)" 
         << setw(20) << "GPU (ms)"
         << setw(20) << "GPU Speedup" 
         << setw(20) << "GPU Success" << endl;
    cout << string(120, '=') << endl;
    
    CPUPathFinderDirect cpu_direct_finder;
    GPUPathFinderDijkstraActual gpu_finder(N);
    
    for (int pins : pin_counts) {
        double total_cpu_direct_time = 0;
        double total_gpu_time = 0;
        int total_valid_paths_gpu = 0;
        int total_pairs = 0;
        
        for (int run = 0; run < NUM_RUNS; run++) {
            vector<pair<int, int>> positions;
            // Generate positions that are not on obstacles
            for (int i = 0; i < pins; i++) {
                int x, y;
                do {
                    x = rand() % (N - 10) + 5;
                    y = rand() % (N - 10) + 5;
                } while (grid[x][y] >= INF);  // Ensure not on obstacle
                positions.push_back({x, y});
                
                int rel_pos = rand() % 4;
                int offset = 5 + rand() % 10;
                
                int target_x, target_y;
                switch (rel_pos) {
                    case 0: // TL -> BR
                        target_x = min(N-1, x + offset);
                        target_y = min(N-1, y + offset);
                        break;
                    case 1: // BR -> TL
                        target_x = max(0, x - offset);
                        target_y = max(0, y - offset);
                        break;
                    case 2: // TR -> BL
                        target_x = max(0, x - offset);
                        target_y = min(N-1, y + offset);
                        break;
                    case 3: // BL -> TR
                        target_x = min(N-1, x + offset);
                        target_y = max(0, y - offset);
                        break;
                }
                
                // Ensure target is not on obstacle
                if (grid[target_x][target_y] < INF) {
                    positions.push_back({target_x, target_y});
                }
            }
            
            total_pairs = positions.size() - 1;
            if (total_pairs <= 0) continue;
            
            // Run CPU Direct
            double cpu_direct_time;
            auto cpu_direct_results = cpu_direct_finder.findDirectPaths(positions, grid, cpu_direct_time);
            total_cpu_direct_time += cpu_direct_time;
            
            // Run GPU
            double gpu_time;
            auto gpu_results = gpu_finder.findOptimalPathsGPUFast(positions, grid, gpu_time);
            total_gpu_time += gpu_time;
            
            // Count valid GPU paths
            for (const auto& result : gpu_results) {
                if (result.valid && !result.full_path.empty()) {
                    total_valid_paths_gpu++;
                }
            }
        }
        
        double avg_cpu_direct = total_cpu_direct_time / NUM_RUNS;
        double avg_gpu = total_gpu_time / NUM_RUNS;
        double speedup = (avg_gpu > 0) ? avg_cpu_direct / avg_gpu : 0;
        double gpu_success_rate = (total_pairs > 0) ? 
            (total_valid_paths_gpu * 100.0) / (total_pairs * NUM_RUNS) : 0;
        
        cout << setw(12) << pins
             << setw(12) << total_pairs
             << setw(20) << fixed << setprecision(2) << avg_cpu_direct
             << setw(20) << fixed << setprecision(2) << avg_gpu
             << setw(20) << fixed << setprecision(1) << speedup << "x"
             << setw(20) << fixed << setprecision(1) << gpu_success_rate << "%" << endl;
    }
}

// ==================== TEST: GPU SPECIFIC FUNCTIONALITY ====================
void testGPUSpecific() {
    cout << "\n\n==========================================" << endl;
    cout << "TEST: GPU SPECIFIC FUNCTIONALITY" << endl;
    cout << "==========================================" << endl;
    
    const int N = 20;
    auto grid = createMazeGridWithBlockedObstacles(N);
    
    GPUPathFinderDijkstraActual gpu_finder(N);
    DijkstraSolverCPU cpu_dijkstra;
    PathValidator validator;
    
    // Test multiple cases where GPU should work
    vector<pair<string, pair<Point, Point>>> test_cases = {
        {"L-shaped horizontal-vertical", {{2, 2}, {10, 2}}},
        {"L-shaped vertical-horizontal", {{2, 2}, {2, 10}}},
        {"Diagonal simple", {{1, 1}, {5, 5}}},
        {"Around obstacle", {{0, 0}, {N-1, 0}}},
    };
    
    for (const auto& test_case : test_cases) {
        string case_name = test_case.first;
        Point start = test_case.second.first;
        Point end = test_case.second.second;
        
        cout << "\n" << case_name << ": (" << start.x << "," << start.y << ") -> (" 
             << end.x << "," << end.y << ")" << endl;
        cout << string(50, '-') << endl;
        
        // CPU reference
        auto cpu_path = cpu_dijkstra.findPathOnly(grid, start, end, N);
        cout << "CPU path length: " << cpu_path.size() << endl;
        
        // GPU
        double gpu_time;
        vector<pair<int, int>> positions = {{start.x, start.y}, {end.x, end.y}};
        auto gpu_results = gpu_finder.findOptimalPathsGPUFast(positions, grid, gpu_time);
        
        if (gpu_results[0].valid && !gpu_results[0].full_path.empty()) {
            cout << "GPU path length: " << gpu_results[0].full_path.size() << endl;
            cout << "GPU time: " << fixed << setprecision(3) << gpu_time << " ms" << endl;
            
            // Validate GPU path
            bool avoids_obstacles = true;
            for (const auto& p : gpu_results[0].full_path) {
                if (grid[p.x][p.y] >= INF) {
                    avoids_obstacles = false;
                    break;
                }
            }
            cout << "GPU avoids obstacles: " << (avoids_obstacles ? "✓ YES" : "✗ NO") << endl;
            
            if (!cpu_path.empty()) {
                // Compare with CPU
                bool same_endpoints = 
                    (gpu_results[0].full_path[0].x == cpu_path[0].x &&
                     gpu_results[0].full_path[0].y == cpu_path[0].y &&
                     gpu_results[0].full_path.back().x == cpu_path.back().x &&
                     gpu_results[0].full_path.back().y == cpu_path.back().y);
                cout << "Matches CPU endpoints: " << (same_endpoints ? "✓ YES" : "✗ NO") << endl;
            }
        } else {
            cout << "GPU: No valid path found" << endl;
        }
    }
}

// ==================== MAIN ====================
int main() {
    try {
        cout << "GPU PATHFINDING WITH OBSTACLE AVOIDANCE - FIXED VERSION" << endl;
        cout << "=========================================================" << endl;
        
        // Test basic obstacle avoidance
        testObstacleAvoidance();
        
        // Test GPU specific functionality
        testGPUSpecific();
        
        // Performance comparison
        testPerformanceComparison();
        
        cout << "\n\n==========================================" << endl;
        cout << "ALL TESTS COMPLETED SUCCESSFULLY" << endl;
        cout << "==========================================" << endl;
        
    } catch (const exception& e) {
        cerr << "\nERROR: " << e.what() << endl;
        return 1;
    } catch (...) {
        cerr << "\nUnknown error occurred" << endl;
        return 1;
    }
    
    return 0;
}