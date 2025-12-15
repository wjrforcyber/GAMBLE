#include "include/mazeRouter.hpp"
#include "include/util.hpp"
//#include "sectionProcessGPU.hpp"
#include <iostream>

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
    // ... other fields as needed
};

// ==================== GPU KERNEL ====================
__global__ void simplePathKernelGPU(
    const int* cost_grid,
    int* path_exists,
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
    
    // Try horizontal then vertical path
    bool path_found = true;
    int step_x = (end_x > start_x) ? 1 : -1;
    for (int x = start_x; x != end_x; x += step_x) {
        if (cost_grid[x * N + start_y] >= INF) {
            path_found = false;
            break;
        }
    }
    
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
        int step_y = (end_y > start_y) ? 1 : -1;
        for (int y = start_y; y != end_y; y += step_y) {
            if (cost_grid[start_x * N + y] >= INF) {
                path_found = false;
                break;
            }
        }
        
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

// ==================== GPU PATH FINDER ====================
class GPUPathFinderDijkstraActual {
private:
    int* d_cost_grid;
    int* d_path_exists;
    int* d_sources_x;
    int* d_sources_y;
    int* d_dests_x;
    int* d_dests_y;
    int grid_size;
    
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
        vector<PathResult> results;
        
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
        
        // Prepare pairs
        vector<int> sources_x, sources_y, dests_x, dests_y;
        
        for (int i = 0; i < num_points - 1; i++) {
            Point source = {positions[i].first, positions[i].second};
            Point sink = {positions[i+1].first, positions[i+1].second};
            
            source = source.clamped(grid_size);
            sink = sink.clamped(grid_size);
            
            sources_x.push_back(source.x);
            sources_y.push_back(source.y);
            dests_x.push_back(sink.x);
            dests_y.push_back(sink.y);
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
            
            simplePathKernelGPU<<<grid_size_kernel, block_size>>>(
                d_cost_grid, d_path_exists, grid_size, num_pairs,
                d_sources_x, d_sources_y, d_dests_x, d_dests_y
            );
            
            cudaDeviceSynchronize();
            
            // Check for errors
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                cerr << "CUDA error: " << cudaGetErrorString(err) << endl;
                // Return empty results
                auto end = high_resolution_clock::now();
                gpu_time_ms = duration_cast<microseconds>(end - start).count() / 1000.0;
                return results;
            }
            
            // Copy results back
            cudaMemcpy(h_path_exists.data(), d_path_exists, sizeof(int) * num_pairs, cudaMemcpyDeviceToHost);
            
            // Process results
            results.resize(num_points - 1);
            for (int p = 0; p < num_pairs; p++) {
                Point source = {positions[p].first, positions[p].second};
                Point sink = {positions[p + 1].first, positions[p + 1].second};
                
                vector<Point> path;
                if (h_path_exists[p] == 1) {
                    // Create L-shaped path
                    path = createLShapedPath(source, sink, cost_grid);
                }
                
                PathResult result;
                result.full_path = path;
                result.valid = !path.empty();
                result.total_cost = calculatePathCost(cost_grid, path);
                results[p] = result;
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

// ==================== MAZE ROUTER GPU INTERFACE ====================
class MazeRouterGPU {
private:
    GPUPathFinderDijkstraActual* gpu_finder;
    int current_grid_size;
    
public:
    MazeRouterGPU() : gpu_finder(nullptr), current_grid_size(0) {}
    
    ~MazeRouterGPU() {
        if (gpu_finder != nullptr) {
            delete gpu_finder;
        }
    }
    
    // Main routing interface
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
            gpu_finder = new GPUPathFinderDijkstraActual(N);
            current_grid_size = N;
        }
        
        // Run GPU path finding
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
        cout << "No path found.\n" << endl;
        // If no paths found but we have pins, at least add the pin positions
        if (gpures.empty() && !pins.empty()) {
            for (const auto& pin : pins) {
                gpures.push_back(pin);
            }
        }
        //remove duplicates
        std::sort(gpures.begin(), gpures.end());
        // Then, use unique to remove consecutive duplicates
        auto it = std::unique(gpures.begin(), gpures.end());
        // Finally, erase the removed elements
        gpures.erase(it, gpures.end());
        
        return gpu_time_ms;
    }
    
    // Additional utility methods
    
    // Route and return detailed path segments
    double routeWithSegments(
        const vector<vector<int>>& cost,
        int N,
        const vector<pair<int, int>>& pins,
        vector<vector<pair<int, int>>>& segments
    ) {
        if (gpu_finder == nullptr || current_grid_size != N) {
            if (gpu_finder != nullptr) delete gpu_finder;
            gpu_finder = new GPUPathFinderDijkstraActual(N);
            current_grid_size = N;
        }
        
        double gpu_time_ms = 0.0;
        auto path_results = gpu_finder->findOptimalPathsGPUFast(pins, cost, gpu_time_ms);
        
        segments.clear();
        for (const auto& result : path_results) {
            vector<pair<int, int>> segment;
            if (result.valid) {
                for (const auto& point : result.full_path) {
                    segment.push_back({point.x, point.y});
                }
            }
            segments.push_back(segment);
        }
        
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
    MazeRouterGPU mazeRouterGPU;
    
    // Check GPU availability
    if (!mazeRouterGPU.isAvailable()) {
        cout << "GPU not available, falling back to CPU" << endl;
        return 1;
    }
    
    cout << "GPU Info: " << mazeRouterGPU.getGPUInfo() << endl;
    
    // Route
    double gputime = mazeRouterGPU.route(costD, N, pins, gpures);
    
    // Print results
    cout << fixed << setprecision(3);
    cout << "GPU routing time: " << gputime << " ms" << endl;
    cout << "Result points: " << gpures.size() << endl;
    assert(checkAllPinsOnPath(gpures, pins));
    cout << "Current GPU version:\n" <<  "    time: " << gputime / 1000 << "s\n    cost: " << evaluate(gpures, cost, N) << endl;

    return 0;
}