//Version old

#include <vector>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cstdlib>
#include <climits>
#include <queue>
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
};

struct DirectPathResult {
    int cost;
    double time_ms;
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

// ==================== CPU DIJKSTRA IMPLEMENTATION ====================
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
    
    DirectPathResult shortestPathDirect(
        const vector<vector<int>>& grid, 
        Point start, 
        Point end, 
        int N
    ) {
        auto start_time = high_resolution_clock::now();
        
        int cost = shortestPath(grid, start, end, N);
        
        auto end_time = high_resolution_clock::now();
        double time_ms = duration_cast<microseconds>(end_time - start_time).count() / 1000.0;
        
        return {cost, time_ms};
    }
};

// ==================== GPU DIJKSTRA KERNEL ====================
__global__ void dijkstraKernel(
    const int* cost_grid,
    int* distances,
    int N,
    int num_pairs,
    int* sources_x,
    int* sources_y,
    int* dests_x,
    int* dests_y
) {
    int pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (pair_idx < num_pairs) {
        int start_x = sources_x[pair_idx];
        int start_y = sources_y[pair_idx];
        int end_x = dests_x[pair_idx];
        int end_y = dests_y[pair_idx];
        
        // Check bounds
        bool start_inside = (start_x >= 0 && start_x < N && start_y >= 0 && start_y < N);
        bool end_inside = (end_x >= 0 && end_x < N && end_y >= 0 && end_y < N);
        
        if (!start_inside) {
            distances[pair_idx] = INF;
            return;
        }
        
        // Simple Dijkstra implementation for GPU
        int* local_dist = new int[N * N];
        bool* local_visited = new bool[N * N];
        
        for (int i = 0; i < N * N; i++) {
            local_dist[i] = INF;
            local_visited[i] = false;
        }
        
        local_dist[start_x * N + start_y] = cost_grid[start_x * N + start_y];
        
        for (int count = 0; count < N * N; count++) {
            int min_dist = INF;
            int min_idx = -1;
            
            for (int i = 0; i < N * N; i++) {
                if (!local_visited[i] && local_dist[i] < min_dist) {
                    min_dist = local_dist[i];
                    min_idx = i;
                }
            }
            
            if (min_idx == -1 || min_dist == INF) break;
            
            int x = min_idx / N;
            int y = min_idx % N;
            local_visited[min_idx] = true;
            
            if (x == end_x && y == end_y) {
                break;
            }
            
            // Update neighbors
            if (x > 0) {
                int idx = (x-1) * N + y;
                int new_dist = local_dist[min_idx] + cost_grid[(x-1) * N + y];
                if (new_dist < local_dist[idx]) {
                    local_dist[idx] = new_dist;
                }
            }
            if (x < N-1) {
                int idx = (x+1) * N + y;
                int new_dist = local_dist[min_idx] + cost_grid[(x+1) * N + y];
                if (new_dist < local_dist[idx]) {
                    local_dist[idx] = new_dist;
                }
            }
            if (y > 0) {
                int idx = x * N + (y-1);
                int new_dist = local_dist[min_idx] + cost_grid[x * N + (y-1)];
                if (new_dist < local_dist[idx]) {
                    local_dist[idx] = new_dist;
                }
            }
            if (y < N-1) {
                int idx = x * N + (y+1);
                int new_dist = local_dist[min_idx] + cost_grid[x * N + (y+1)];
                if (new_dist < local_dist[idx]) {
                    local_dist[idx] = new_dist;
                }
            }
        }
        
        distances[pair_idx] = end_inside ? local_dist[end_x * N + end_y] : INF;
        
        delete[] local_dist;
        delete[] local_visited;
    }
}

// ==================== CPU DIRECT PATH FINDER (NO DIAGONAL BRIDGING) ====================
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

// ==================== CPU PATH FINDER WITH DIAGONAL BRIDGING ====================
class CPUPathFinderDiagonal {
private:
    DijkstraSolverCPU dijkstra_solver;
    
    PathResult computeDiagonalBridging(
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
            int cost = dijkstra_solver.shortestPath(grid, source, sink, N);
            result.valid = true;
            result.total_cost = cost;
            result.diagonal_point = source;
            result.cost_source_to_d = 0;
            result.cost_d_to_sink = cost;
            result.square_size = 1;
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
        
        int min_total_cost = INF;
        int optimal_j = -1;
        int cost_to_d = 0;
        int cost_from_d = 0;
        Point optimal_d_normalized;
        
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
            
            int cost_sd = dijkstra_solver.shortestPath(grid, trans_info.normalized_source, d_normalized, N);
            int cost_dt = dijkstra_solver.shortestPath(grid, d_normalized, trans_info.normalized_sink, N);
            
            if (cost_sd >= INF || cost_dt >= INF) {
                continue;
            }
            
            int total_cost = cost_sd + cost_dt;
            
            if (total_cost < min_total_cost) {
                min_total_cost = total_cost;
                optimal_j = j;
                cost_to_d = cost_sd;
                cost_from_d = cost_dt;
                optimal_d_normalized = d_normalized;
            }
        }
        
        if (optimal_j != -1 && min_total_cost < INF) {
            // Transform diagonal point back to original coordinates
            Point optimal_d_original = transformBack(optimal_d_normalized, trans_info, N);
            
            // Adjust costs if swapped
            if (trans_info.swapped) {
                swap(cost_to_d, cost_from_d);
            }
            
            result = {
                optimal_d_original,
                cost_to_d,
                cost_from_d,
                min_total_cost,
                n,
                true,
                exceeds_boundary,
                trans_info.flip_x,
                trans_info.flip_y,
                trans_info.swapped,
                trans_info.case_type
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
            PathResult result = computeDiagonalBridging(cost_grid, source, sink, N, trans_info);
            
            results.push_back(result);
        }
        
        auto end = high_resolution_clock::now();
        cpu_time_ms = duration_cast<microseconds>(end - start).count() / 1000.0;
        
        return results;
    }
};

// ==================== GPU PATH FINDER ====================
class GPUPathFinderDijkstraActual {
private:
    int* d_cost_grid;
    int* d_distances;
    int* d_sources_x;
    int* d_sources_y;
    int* d_dests_x;
    int* d_dests_y;
    int grid_size;
    
public:
    GPUPathFinderDijkstraActual(int N) : grid_size(N) {
        cudaMalloc(&d_cost_grid, sizeof(int) * N * N);
        cudaMalloc(&d_distances, sizeof(int) * 1000000);
        cudaMalloc(&d_sources_x, sizeof(int) * 1000000);
        cudaMalloc(&d_sources_y, sizeof(int) * 1000000);
        cudaMalloc(&d_dests_x, sizeof(int) * 1000000);
        cudaMalloc(&d_dests_y, sizeof(int) * 1000000);
    }
    
    ~GPUPathFinderDijkstraActual() {
        cudaFree(d_cost_grid);
        cudaFree(d_distances);
        cudaFree(d_sources_x);
        cudaFree(d_sources_y);
        cudaFree(d_dests_x);
        cudaFree(d_dests_y);
    }
    
    vector<PathResult> findOptimalPathsGPU(
        const vector<pair<int, int>>& positions,
        const vector<vector<int>>& cost_grid,
        double& gpu_time_ms
    ) {
        auto start = high_resolution_clock::now();
        
        int num_points = positions.size();
        if (num_points < 2) {
            gpu_time_ms = 0;
            return vector<PathResult>(num_points - 1);
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
        
        // Prepare all pairs for GPU computation
        vector<int> sources_x, sources_y, dests_x, dests_y;
        vector<int> pair_indices;
        vector<int> pair_j_values;
        vector<TransformationInfo> trans_infos;
        
        for (int i = 0; i < num_points - 1; i++) {
            Point source = {positions[i].first, positions[i].second};
            Point sink = {positions[i+1].first, positions[i+1].second};
            
            TransformationInfo trans_info = normalizePoints(source, sink, grid_size);
            
            // Skip cases not suitable for diagonal bridging
            if (trans_info.case_type == 5 || trans_info.case_type == 6) {
                continue;
            }
            
            int dr = trans_info.normalized_sink.x - trans_info.normalized_source.x;
            int dc = trans_info.normalized_sink.y - trans_info.normalized_source.y;
            int n = max(dr, dc) + 1;
            
            for (int j = 0; j < n; j++) {
                Point d_normalized = {
                    trans_info.normalized_source.x + j,
                    trans_info.normalized_source.y + (n - 1 - j)
                };
                
                // Skip points outside grid
                if (d_normalized.x < 0 || d_normalized.x >= grid_size ||
                    d_normalized.y < 0 || d_normalized.y >= grid_size) {
                    continue;
                }
                
                // Source to D
                sources_x.push_back(trans_info.normalized_source.x);
                sources_y.push_back(trans_info.normalized_source.y);
                dests_x.push_back(d_normalized.x);
                dests_y.push_back(d_normalized.y);
                pair_indices.push_back(i);
                pair_j_values.push_back(j);
                trans_infos.push_back(trans_info);
                
                // D to Sink
                sources_x.push_back(d_normalized.x);
                sources_y.push_back(d_normalized.y);
                dests_x.push_back(trans_info.normalized_sink.x);
                dests_y.push_back(trans_info.normalized_sink.y);
                pair_indices.push_back(i);
                pair_j_values.push_back(j);
                trans_infos.push_back(trans_info);
            }
        }
        
        int num_pairs = sources_x.size();
        vector<PathResult> results(num_points - 1);
        
        // Initialize results
        for (int i = 0; i < num_points - 1; i++) {
            results[i].valid = false;
            results[i].total_cost = INF;
        }
        
        if (num_pairs > 0) {
            // Copy data to GPU
            cudaMemcpy(d_sources_x, sources_x.data(), sizeof(int) * num_pairs, cudaMemcpyHostToDevice);
            cudaMemcpy(d_sources_y, sources_y.data(), sizeof(int) * num_pairs, cudaMemcpyHostToDevice);
            cudaMemcpy(d_dests_x, dests_x.data(), sizeof(int) * num_pairs, cudaMemcpyHostToDevice);
            cudaMemcpy(d_dests_y, dests_y.data(), sizeof(int) * num_pairs, cudaMemcpyHostToDevice);
            
            vector<int> h_distances(num_pairs);
            
            // Launch kernel
            int block_size = 256;
            int grid_size_kernel = (num_pairs + block_size - 1) / block_size;
            
            dijkstraKernel<<<grid_size_kernel, block_size>>>(
                d_cost_grid, d_distances, grid_size, num_pairs, 
                d_sources_x, d_sources_y, d_dests_x, d_dests_y
            );
            cudaDeviceSynchronize();
            
            // Copy results back
            cudaMemcpy(h_distances.data(), d_distances, sizeof(int) * num_pairs, cudaMemcpyDeviceToHost);
            
            // Process results
            vector<vector<pair<int, int>>> pair_costs(num_points - 1);
            vector<vector<Point>> diagonal_points(num_points - 1);
            
            for (int p = 0; p < num_pairs; p += 2) {
                int result_idx = pair_indices[p];
                int j = pair_j_values[p];
                TransformationInfo info = trans_infos[p];
                
                int cost_sd = h_distances[p];
                int cost_dt = h_distances[p + 1];
                
                if (cost_sd < INF/2 && cost_dt < INF/2) {
                    int total_cost = cost_sd + cost_dt;
                    pair_costs[result_idx].push_back({total_cost, j});
                    
                    Point d_normalized = {dests_x[p], dests_y[p]};
                    diagonal_points[result_idx].push_back(d_normalized);
                }
            }
            
            // Find optimal paths
            for (int i = 0; i < num_points - 1; i++) {
                if (!pair_costs[i].empty()) {
                    Point source = {positions[i].first, positions[i].second};
                    Point sink = {positions[i+1].first, positions[i+1].second};
                    
                    TransformationInfo trans_info = normalizePoints(source, sink, grid_size);
                    
                    int min_cost = INF;
                    int optimal_j = -1;
                    for (size_t k = 0; k < pair_costs[i].size(); k++) {
                        if (pair_costs[i][k].first < min_cost) {
                            min_cost = pair_costs[i][k].first;
                            optimal_j = pair_costs[i][k].second;
                        }
                    }
                    
                    if (optimal_j != -1) {
                        Point optimal_d_normalized = diagonal_points[i][optimal_j];
                        Point optimal_d_original = transformBack(optimal_d_normalized, trans_info, grid_size);
                        
                        bool exceeds_x = (trans_info.normalized_source.x + trans_info.normalized_sink.x) > grid_size;
                        bool exceeds_y = (trans_info.normalized_source.y + trans_info.normalized_sink.y) > grid_size;
                        bool exceeds_boundary = exceeds_x || exceeds_y;
                        
                        results[i] = {
                            optimal_d_original,
                            0, // cost_to_d
                            0, // cost_from_d
                            min_cost,
                            (int)pair_costs[i].size() / 2 + 1,
                            true,
                            exceeds_boundary,
                            trans_info.flip_x,
                            trans_info.flip_y,
                            trans_info.swapped,
                            trans_info.case_type
                        };
                    }
                }
            }
        }
        
        // Handle cases not processed by GPU (colinear, same point, etc.)
        DijkstraSolverCPU cpu_solver;
        for (int i = 0; i < num_points - 1; i++) {
            if (!results[i].valid) {
                Point source = {positions[i].first, positions[i].second};
                Point sink = {positions[i+1].first, positions[i+1].second};
                
                TransformationInfo trans_info = normalizePoints(source, sink, grid_size);
                int direct_cost = cpu_solver.shortestPath(cost_grid, source, sink, grid_size);
                
                results[i] = {
                    source,
                    0,
                    0,
                    direct_cost,
                    1,
                    true,
                    false,
                    trans_info.flip_x,
                    trans_info.flip_y,
                    trans_info.swapped,
                    trans_info.case_type
                };
            }
        }
        
        auto end = high_resolution_clock::now();
        gpu_time_ms = duration_cast<microseconds>(end - start).count() / 1000.0;
        
        return results;
    }
};

// ==================== TEST FUNCTIONS ====================
vector<vector<int>> createMazeGrid(int N) {
    vector<vector<int>> grid(N, vector<int>(N, 1));
    
    // Create some obstacles
    for (int i = N/4; i < 3*N/4; i++) {
        grid[N/2][i] = 1000;
        grid[i][N/2] = 1000;
    }
    
    // Create a cheap path
    for (int i = 0; i < N; i++) {
        grid[i][i] = 1;
    }
    
    // Add some random variations
    srand(42);
    for (int i = 0; i < N; i += 3) {
        for (int j = 0; j < N; j += 3) {
            if (rand() % 3 == 0) {
                grid[i][j] = 50 + rand() % 50;
            }
        }
    }
    
    return grid;
}

// ==================== TEST: ALL CASES ====================
void testAllCases() {
    cout << "\n\n==========================================" << endl;
    cout << "TEST: ALL RELATIVE POSITIONS" << endl;
    cout << "==========================================" << endl;
    
    const int N = 25;
    auto grid = createMazeGrid(N);
    DijkstraSolverCPU dijkstra;
    
    vector<pair<string, pair<Point, Point>>> test_cases = {
        {"Case 1: TL->BR", {{2, 2}, {20, 20}}},
        {"Case 2: BR->TL", {{20, 20}, {2, 2}}},
        {"Case 3: TR->BL", {{20, 2}, {2, 20}}},
        {"Case 4: BL->TR", {{2, 20}, {20, 2}}},
        {"Case 5: Horizontal", {{10, 10}, {20, 10}}},
        {"Case 6: Vertical", {{10, 10}, {10, 20}}},
        {"Case 7: Same point", {{15, 15}, {15, 15}}}
    };
    
    cout << "\nTesting normalizePoints function:" << endl;
    cout << string(80, '=') << endl;
    cout << setw(25) << "Case"
         << setw(20) << "Source->Sink"
         << setw(15) << "Case Type"
         << setw(20) << "Transformations" << endl;
    cout << string(80, '=') << endl;
    
    for (const auto& test_case : test_cases) {
        string case_name = test_case.first;
        Point source = test_case.second.first;
        Point sink = test_case.second.second;
        
        TransformationInfo info = normalizePoints(source, sink, N);
        
    cout << setw(25) << case_name 
         << setw(20) << "(" << source.x << "," << source.y << ")" << "->(" << sink.x << "," << sink.y << ")"
         << setw(15) << info.case_type
         << setw(20);
        
        if (info.flip_x) cout << "FlipX ";
        if (info.flip_y) cout << "FlipY ";
        if (info.swapped) cout << "Swapped ";
        if (!info.flip_x && !info.flip_y && !info.swapped) cout << "None";
        cout << endl;
    }
    
    // Test the path finder
    cout << "\n\nTesting CPU Path Finder with mixed sequence:" << endl;
    cout << string(60, '=') << endl;
    
    vector<pair<int, int>> positions = {
        {2, 2}, {20, 20}, {5, 5}, {20, 5}, {5, 20}, {10, 10}, {15, 10}, {10, 15}, {10, 10}, {2, 2}
    };
    
    CPUPathFinderDiagonal path_finder;
    double cpu_time;
    auto results = path_finder.findOptimalPaths(positions, grid, cpu_time);
    
    cout << "Processed " << positions.size() << " positions in " 
         << fixed << setprecision(2) << cpu_time << " ms" << endl;
    
    int valid_count = 0;
    for (size_t i = 0; i < results.size(); i++) {
        if (results[i].valid) {
            valid_count++;
        }
    }
    cout << "Valid paths found: " << valid_count << "/" << results.size() << endl;
}

// ==================== TEST: PERFORMANCE ====================
void testPerformanceComparison() {
    cout << "\n\n==========================================" << endl;
    cout << "TEST: PERFORMANCE COMPARISON" << endl;
    cout << "==========================================" << endl;
    
    const int N = 50;
    const int NUM_RUNS = 2;
    
    auto grid = createMazeGrid(N);
    
    vector<int> pin_counts = {5, 10, 20, 100};
    
    cout << fixed << setprecision(3);
    cout << "\nPerformance (Grid: " << N << "x" << N << ")" << endl;
    cout << "\n" << string(100, '=') << endl;
    cout << setw(12) << "Pins" 
         << setw(12) << "Pairs"
         << setw(20) << "CPU Direct (ms)" 
         << setw(20) << "CPU Diagonal (ms)"
         << setw(20) << "GPU Diagonal (ms)" << endl;
    cout << string(100, '=') << endl;
    
    CPUPathFinderDirect cpu_direct_finder;
    CPUPathFinderDiagonal cpu_diagonal_finder;
    GPUPathFinderDijkstraActual gpu_finder(N);
    
    for (int pins : pin_counts) {
        double total_cpu_direct_time = 0;
        double total_cpu_diagonal_time = 0;
        double total_gpu_time = 0;
        int total_pairs = 0;
        
        for (int run = 0; run < NUM_RUNS; run++) {
            vector<pair<int, int>> positions;
            for (int i = 0; i < pins; i++) {
                int x = rand() % (N - 10);
                int y = rand() % (N - 10);
                positions.push_back({x, y});
                
                // Add next point with random relative position
                int rel_pos = rand() % 4;
                int offset = 5 + rand() % 10;
                
                switch (rel_pos) {
                    case 0: // TL -> BR
                        positions.push_back({min(N-1, x + offset), min(N-1, y + offset)});
                        break;
                    case 1: // BR -> TL
                        positions.push_back({max(0, x - offset), max(0, y - offset)});
                        break;
                    case 2: // TR -> BL
                        positions.push_back({max(0, x - offset), min(N-1, y + offset)});
                        break;
                    case 3: // BL -> TR
                        positions.push_back({min(N-1, x + offset), max(0, y - offset)});
                        break;
                }
            }
            
            total_pairs = positions.size() - 1;
            
            // Run CPU Direct
            double cpu_direct_time;
            auto cpu_direct_results = cpu_direct_finder.findDirectPaths(positions, grid, cpu_direct_time);
            total_cpu_direct_time += cpu_direct_time;
            
            // Run CPU Diagonal
            double cpu_diagonal_time;
            auto cpu_diagonal_results = cpu_diagonal_finder.findOptimalPaths(positions, grid, cpu_diagonal_time);
            total_cpu_diagonal_time += cpu_diagonal_time;
            
            // Run GPU
            double gpu_time;
            auto gpu_results = gpu_finder.findOptimalPathsGPU(positions, grid, gpu_time);
            total_gpu_time += gpu_time;
        }
        
        double avg_cpu_direct = total_cpu_direct_time / NUM_RUNS;
        double avg_cpu_diagonal = total_cpu_diagonal_time / NUM_RUNS;
        double avg_gpu = total_gpu_time / NUM_RUNS;
        
        cout << setw(12) << pins
             << setw(12) << total_pairs
             << setw(20) << fixed << setprecision(2) << avg_cpu_direct
             << setw(20) << fixed << setprecision(2) << avg_cpu_diagonal
             << setw(20) << fixed << setprecision(2) << avg_gpu << endl;
    }
}

// ==================== TEST: ACCURACY ====================
void testAccuracy() {
    cout << "\n\n==========================================" << endl;
    cout << "TEST: ACCURACY VERIFICATION" << endl;
    cout << "==========================================" << endl;
    
    const int N = 20;
    auto grid = createMazeGrid(N);
    
    DijkstraSolverCPU dijkstra;
    CPUPathFinderDiagonal path_finder;
    
    // Test cases
    vector<pair<Point, Point>> test_pairs = {
        {{0, 0}, {N-1, N-1}},
        {{5, 5}, {15, 15}},
        {{15, 5}, {5, 15}},
        {{5, 15}, {15, 5}},
        {{10, 0}, {10, N-1}},
        {{0, 10}, {N-1, 10}}
    };
    
    cout << "\nComparing Direct Dijkstra vs Diagonal Bridging:" << endl;
    cout << string(80, '=') << endl;
    cout << setw(25) << "Points"
         << setw(15) << "Direct"
         << setw(15) << "Diagonal" << endl;
    cout << string(80, '=') << endl;
    
    bool all_match = true;
    for (const auto& pair : test_pairs) {
        Point source = pair.first;
        Point sink = pair.second;
        
        // Direct Dijkstra
        int direct_cost = dijkstra.shortestPath(grid, source, sink, N);
        
        // Diagonal Bridging
        vector<std::pair<int, int>> positions = {{source.x, source.y}, {sink.x, sink.y}};
        double time;
        auto results = path_finder.findOptimalPaths(positions, grid, time);
        int diagonal_cost = results[0].valid ? results[0].total_cost : INF;

        cout << setw(25) << "(" << source.x << "," << source.y << ")" << "->(" << sink.x << "," << sink.y << ")"
             << setw(15) << direct_cost
             << setw(15) << diagonal_cost << endl;
    }
}

// ==================== MAIN ====================
int main() {
    cout << "==========================================" << endl;
    cout << "GENERALIZED PATH FINDER" << endl;
    cout << "==========================================" << endl;
    cout << "Handles all relative positions:" << endl;
    cout << "1. TL->BR (Case 1)" << endl;
    cout << "2. BR->TL (Case 2)" << endl;
    cout << "3. TR->BL (Case 3)" << endl;
    cout << "4. BL->TR (Case 4)" << endl;
    cout << "5. Colinear (Case 5)" << endl;
    cout << "6. Same point (Case 6)" << endl;
    cout << "==========================================" << endl;
    
    try {
        testAllCases();
        testPerformanceComparison();
        testAccuracy();
    } catch (const exception& e) {
        cerr << "\nERROR: " << e.what() << endl;
        return 1;
    } catch (...) {
        cerr << "\nUnknown error occurred" << endl;
        return 1;
    }
    
    return 0;
}