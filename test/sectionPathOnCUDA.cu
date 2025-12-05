#include <vector>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cstdlib>
#include <climits>
#include <cuda_runtime.h>

using namespace std;
using namespace std::chrono;

// ==================== DATA STRUCTURES ====================
struct Point {
    int x, y;
};

struct PathResult {
    Point diagonal_point;
    int cost_source_to_d;
    int cost_d_to_sink;
    int total_cost;
    int square_size;
    bool valid;
};

// ==================== GPU KERNEL ====================
__global__ void findOptimalPathsKernel(
    const Point* points,
    const int* cost_grid,
    PathResult* results,
    int N,
    int num_pairs
) {
    int pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (pair_idx < num_pairs) {
        Point source = points[pair_idx];
        Point sink = points[pair_idx + 1];
        
        // Initialize as invalid
        results[pair_idx].valid = false;
        results[pair_idx].total_cost = INT_MAX;
        
        // Check if source is upper-left of sink
        if (source.x <= sink.x && source.y <= sink.y) {
            int dr = sink.x - source.x;
            int dc = sink.y - source.y;
            
            int n = max(dr, dc) + 1;
            n = min(n, N - source.x);
            n = min(n, N - source.y);
            n = max(n, 1);
            
            int min_total_cost = INT_MAX;
            int optimal_i = -1;
            int cost_to_d = 0;
            int cost_from_d = 0;
            
            // For each point on second diagonal
            for (int i = 0; i < n; i++) {
                Point d = {source.x + i, source.y + (n - 1 - i)};
                
                // Compute cost from source to d
                int cost_sd = 0;
                // Horizontal then vertical path
                for (int x = source.x; x < d.x; x++) {
                    cost_sd += cost_grid[x * N + source.y];
                }
                for (int y = source.y; y <= d.y; y++) {
                    cost_sd += cost_grid[d.x * N + y];
                }
                // Subtract source point (counted twice)
                cost_sd -= cost_grid[source.x * N + source.y];
                
                // Compute cost from d to sink
                int cost_dt = 0;
                for (int x = d.x; x < sink.x; x++) {
                    cost_dt += cost_grid[x * N + d.y];
                }
                for (int y = d.y; y <= sink.y; y++) {
                    cost_dt += cost_grid[sink.x * N + y];
                }
                // Subtract d point (counted twice)
                cost_dt -= cost_grid[d.x * N + d.y];
                
                int total_cost = cost_sd + cost_dt;
                
                if (total_cost < min_total_cost) {
                    min_total_cost = total_cost;
                    optimal_i = i;
                    cost_to_d = cost_sd;
                    cost_from_d = cost_dt;
                }
            }
            
            if (optimal_i != -1) {
                Point optimal_d = {source.x + optimal_i, source.y + (n - 1 - optimal_i)};
                results[pair_idx] = {
                    optimal_d,
                    cost_to_d,
                    cost_from_d,
                    min_total_cost,
                    n,
                    true
                };
            }
        }
    }
}

// ==================== GPU PATH FINDER ====================
class GPUPathFinder {
private:
    Point* d_points;
    int* d_cost_grid;
    PathResult* d_results;
    int grid_size;
    
public:
    GPUPathFinder(int N) : grid_size(N) {
        // Allocate with error checking
        cudaError_t err;
        
        err = cudaMalloc(&d_points, sizeof(Point) * 1000000);
        if (err != cudaSuccess) {
            cerr << "Failed to allocate d_points: " << cudaGetErrorString(err) << endl;
            exit(1);
        }
        
        err = cudaMalloc(&d_cost_grid, sizeof(int) * N * N);
        if (err != cudaSuccess) {
            cerr << "Failed to allocate d_cost_grid: " << cudaGetErrorString(err) << endl;
            cudaFree(d_points);
            exit(1);
        }
        
        err = cudaMalloc(&d_results, sizeof(PathResult) * 1000000);
        if (err != cudaSuccess) {
            cerr << "Failed to allocate d_results: " << cudaGetErrorString(err) << endl;
            cudaFree(d_points);
            cudaFree(d_cost_grid);
            exit(1);
        }
    }
    
    ~GPUPathFinder() {
        cudaFree(d_points);
        cudaFree(d_cost_grid);
        cudaFree(d_results);
    }
    
    vector<PathResult> findOptimalPaths(
        const vector<pair<int, int>>& positions,
        const vector<vector<int>>& cost_grid,
        double& gpu_time_ms
    ) {
        auto start = high_resolution_clock::now();
        
        int num_points = positions.size();
        if (num_points < 2) {
            gpu_time_ms = 0;
            return {};
        }
        
        int num_pairs = num_points - 1;
        
        // Convert points
        vector<Point> h_points(num_points);
        for (int i = 0; i < num_points; i++) {
            h_points[i] = {positions[i].first, positions[i].second};
        }
        
        // Flatten cost grid
        vector<int> flat_cost(grid_size * grid_size);
        for (int i = 0; i < grid_size; i++) {
            for (int j = 0; j < grid_size; j++) {
                flat_cost[i * grid_size + j] = cost_grid[i][j];
            }
        }
        
        // Copy to GPU
        cudaMemcpy(d_points, h_points.data(), 
                  sizeof(Point) * num_points, cudaMemcpyHostToDevice);
        cudaMemcpy(d_cost_grid, flat_cost.data(),
                  sizeof(int) * grid_size * grid_size, cudaMemcpyHostToDevice);
        
        // Launch kernel
        int block_size = 256;
        int num_blocks = (num_pairs + block_size - 1) / block_size;
        
        findOptimalPathsKernel<<<num_blocks, block_size>>>(
            d_points, d_cost_grid, d_results, grid_size, num_pairs
        );
        
        // Check for errors
        cudaError_t kernel_err = cudaGetLastError();
        if (kernel_err != cudaSuccess) {
            cerr << "Kernel error: " << cudaGetErrorString(kernel_err) << endl;
            gpu_time_ms = 0;
            return {};
        }
        
        cudaDeviceSynchronize();
        
        // Get results
        vector<PathResult> results(num_pairs);
        cudaMemcpy(results.data(), d_results,
                  sizeof(PathResult) * num_pairs, cudaMemcpyDeviceToHost);
        
        auto end = high_resolution_clock::now();
        gpu_time_ms = duration_cast<microseconds>(end - start).count() / 1000.0;
        
        return results;
    }
};

// ==================== CPU PATH FINDER ====================
class CPUPathFinder {
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
            
            PathResult result;
            result.valid = false;
            result.total_cost = INT_MAX;
            
            if (source.x <= sink.x && source.y <= sink.y) {
                int dr = sink.x - source.x;
                int dc = sink.y - source.y;
                
                int n = max(dr, dc) + 1;
                n = min(n, N - source.x);
                n = min(n, N - source.y);
                n = max(n, 1);
                
                int min_total_cost = INT_MAX;
                int optimal_i = -1;
                int cost_to_d = 0;
                int cost_from_d = 0;
                
                for (int j = 0; j < n; j++) {
                    Point d = {source.x + j, source.y + (n - 1 - j)};
                    
                    // Compute cost from source to d
                    int cost_sd = 0;
                    for (int x = source.x; x < d.x; x++) {
                        cost_sd += cost_grid[x][source.y];
                    }
                    for (int y = source.y; y <= d.y; y++) {
                        cost_sd += cost_grid[d.x][y];
                    }
                    cost_sd -= cost_grid[source.x][source.y];
                    
                    // Compute cost from d to sink
                    int cost_dt = 0;
                    for (int x = d.x; x < sink.x; x++) {
                        cost_dt += cost_grid[x][d.y];
                    }
                    for (int y = d.y; y <= sink.y; y++) {
                        cost_dt += cost_grid[sink.x][y];
                    }
                    cost_dt -= cost_grid[d.x][d.y];
                    
                    int total_cost = cost_sd + cost_dt;
                    
                    if (total_cost < min_total_cost) {
                        min_total_cost = total_cost;
                        optimal_i = j;
                        cost_to_d = cost_sd;
                        cost_from_d = cost_dt;
                    }
                }
                
                if (optimal_i != -1) {
                    Point optimal_d = {source.x + optimal_i, source.y + (n - 1 - optimal_i)};
                    result = {
                        optimal_d,
                        cost_to_d,
                        cost_from_d,
                        min_total_cost,
                        n,
                        true
                    };
                }
            }
            
            results.push_back(result);
        }
        
        auto end = high_resolution_clock::now();
        cpu_time_ms = duration_cast<microseconds>(end - start).count() / 1000.0;
        
        return results;
    }
};

// ==================== UTILITY FUNCTIONS ====================
vector<vector<int>> generateRandomCostGrid(int N, int min_cost = 1, int max_cost = 100) {
    vector<vector<int>> grid(N, vector<int>(N));
    
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            grid[i][j] = min_cost + rand() % (max_cost - min_cost + 1);
        }
    }
    
    return grid;
}

vector<pair<int, int>> generateDiagonalPositions(int N, int num_points, int max_step = 10) {
    vector<pair<int, int>> positions;
    int x = 0, y = 0;
    
    for (int i = 0; i < num_points; i++) {
        positions.push_back({x, y});
        
        int step_x = 1 + rand() % min(max_step, N - x - 1);
        int step_y = 1 + rand() % min(max_step, N - y - 1);
        
        x = min(N - 1, x + step_x);
        y = min(N - 1, y + step_y);
        
        // Reset if we hit boundary
        if (x >= N - 5 || y >= N - 5) {
            x = rand() % (N / 2);
            y = rand() % (N / 2);
        }
    }
    
    return positions;
}

void printPerformanceHeader() {
    cout << "\n" << string(120, '=') << endl;
    cout << setw(12) << "Pins" 
         << setw(12) << "Pairs"
         << setw(12) << "Valid"
         << setw(15) << "Avg Square"
         << setw(15) << "CPU (ms)" 
         << setw(15) << "GPU (ms)"
         << setw(15) << "Speedup" 
         << setw(15) << "Winner"
         << setw(15) << "GPU Eff%" << endl;
    cout << string(120, '=') << endl;
}

void printPerformanceRow(int pins, int pairs, int valid_pairs, double avg_square,
                         double cpu_time, double gpu_time, double speedup) {
    string winner = (speedup > 1.0) ? "GPU" : "CPU";
    double gpu_efficiency = (speedup > 1.0) ? (speedup - 1.0) / speedup * 100 : 0;
    
    cout << fixed << setprecision(3);
    cout << setw(12) << pins
         << setw(12) << pairs
         << setw(12) << valid_pairs
         << setw(15) << avg_square
         << setw(15) << cpu_time
         << setw(15) << gpu_time
         << setw(15) << (speedup > 1.0 ? speedup : 1.0/speedup)
         << setw(15) << winner
         << setw(15) << fixed << setprecision(1) << gpu_efficiency << "%" << endl;
}

// ==================== PERFORMANCE COMPARISON ACROSS PIN COUNTS ====================
void runPinCountComparison() {
    cout << "\n==========================================" << endl;
    cout << "PERFORMANCE COMPARISON: DIFFERENT PIN COUNTS" << endl;
    cout << "==========================================" << endl;
    
    const int N = 1000;  // Large grid
    const int NUM_RUNS = 5;
    
    // Test different numbers of pins
    vector<int> pin_counts = {10, 50, 100, 500, 1000, 5000, 10000, 50000};
    
    // Generate cost grid once
    auto cost_grid = generateRandomCostGrid(N, 1, 100);
    
    CPUPathFinder cpu_finder;
    GPUPathFinder gpu_finder(N);
    
    printPerformanceHeader();
    
    for (int num_pins : pin_counts) {
        double total_cpu_time = 0;
        double total_gpu_time = 0;
        int total_valid_pairs = 0;
        double total_avg_square = 0;
        int total_pairs = 0;
        
        for (int run = 0; run < NUM_RUNS; run++) {
            // Generate positions for this run
            auto positions = generateDiagonalPositions(N, num_pins, 50);
            
            // CPU execution
            double cpu_time;
            auto cpu_results = cpu_finder.findOptimalPaths(positions, cost_grid, cpu_time);
            total_cpu_time += cpu_time;
            
            // GPU execution
            double gpu_time;
            auto gpu_results = gpu_finder.findOptimalPaths(positions, cost_grid, gpu_time);
            total_gpu_time += gpu_time;
            
            // Collect statistics on first run
            if (run == 0) {
                total_pairs = cpu_results.size();
                
                int valid_count = 0;
                double square_sum = 0;
                for (const auto& res : cpu_results) {
                    if (res.valid) {
                        valid_count++;
                        square_sum += res.square_size;
                    }
                }
                total_valid_pairs = valid_count;
                total_avg_square = (valid_count > 0) ? square_sum / valid_count : 0;
            }
            
            // Quick validation
            if (cpu_results.size() != gpu_results.size()) {
                cout << "  Warning: Size mismatch for " << num_pins << " pins, run " << run << endl;
            }
        }
        
        // Calculate averages
        double avg_cpu_time = total_cpu_time / NUM_RUNS;
        double avg_gpu_time = total_gpu_time / NUM_RUNS;
        double speedup = avg_cpu_time / avg_gpu_time;
        
        printPerformanceRow(num_pins, total_pairs, total_valid_pairs, total_avg_square,
                           avg_cpu_time, avg_gpu_time, speedup);
        
        // Break if taking too long
        if (avg_cpu_time > 10000) { // 10 seconds
            cout << "\nStopping test - CPU time > 10 seconds" << endl;
            break;
        }
    }
}

// ==================== BREAK-EVEN ANALYSIS ====================
void analyzeBreakEvenPoint() {
    cout << "\n\n==========================================" << endl;
    cout << "BREAK-EVEN ANALYSIS: WHEN GPU BECOMES FASTER" << endl;
    cout << "==========================================" << endl;
    
    const int N = 1000;
    const int NUM_RUNS = 3;
    
    auto cost_grid = generateRandomCostGrid(N, 1, 100);
    CPUPathFinder cpu_finder;
    GPUPathFinder gpu_finder(N);
    
    cout << "\nFinding break-even point (where GPU becomes faster than CPU):" << endl;
    cout << "Grid size: " << N << "x" << N << endl;
    cout << "\n" << string(80, '-') << endl;
    cout << setw(10) << "Pins" 
         << setw(15) << "CPU (ms)" 
         << setw(15) << "GPU (ms)"
         << setw(15) << "Speedup" 
         << setw(20) << "Status" << endl;
    cout << string(80, '-') << endl;
    
    // Test increasing pin counts to find break-even
    for (int pins = 10; pins <= 100000; pins *= 2) {
        double total_cpu_time = 0;
        double total_gpu_time = 0;
        
        for (int run = 0; run < NUM_RUNS; run++) {
            auto positions = generateDiagonalPositions(N, pins, 30);
            
            double cpu_time, gpu_time;
            cpu_finder.findOptimalPaths(positions, cost_grid, cpu_time);
            gpu_finder.findOptimalPaths(positions, cost_grid, gpu_time);
            
            total_cpu_time += cpu_time;
            total_gpu_time += gpu_time;
        }
        
        double avg_cpu = total_cpu_time / NUM_RUNS;
        double avg_gpu = total_gpu_time / NUM_RUNS;
        double speedup = avg_cpu / avg_gpu;
        
        cout << fixed << setprecision(3);
        cout << setw(10) << pins
             << setw(15) << avg_cpu
             << setw(15) << avg_gpu
             << setw(15) << (speedup > 1.0 ? speedup : 1.0/speedup);
        
        if (speedup > 1.0) {
            cout << setw(20) << "GPU FASTER ✓" << endl;
            cout << "\n✓ Break-even point found at ~" << pins << " pins" << endl;
            break;
        } else {
            cout << setw(20) << "CPU faster" << endl;
        }
        
        if (pins >= 100000) {
            cout << "\n✗ GPU never became faster (tested up to 100,000 pins)" << endl;
        }
    }
}

// ==================== THROUGHPUT ANALYSIS ====================
void analyzeThroughput() {
    cout << "\n\n==========================================" << endl;
    cout << "THROUGHPUT ANALYSIS: OPERATIONS PER SECOND" << endl;
    cout << "==========================================" << endl;
    
    const int N = 500;
    const int NUM_RUNS = 3;
    
    auto cost_grid = generateRandomCostGrid(N, 1, 100);
    CPUPathFinder cpu_finder;
    GPUPathFinder gpu_finder(N);
    
    vector<int> test_pins = {100, 1000, 10000, 50000};
    
    cout << "\nThroughput in million operations per second:" << endl;
    cout << "Operation = processing one diagonal point for one pair" << endl;
    cout << "\n" << string(100, '=') << endl;
    cout << setw(12) << "Pins"
         << setw(12) << "Pairs"
         << setw(15) << "Ops/Pair"
         << setw(15) << "Total Ops"
         << setw(15) << "CPU Mops/s"
         << setw(15) << "GPU Mops/s"
         << setw(15) << "Ratio" << endl;
    cout << string(100, '=') << endl;
    
    for (int pins : test_pins) {
        double total_cpu_time = 0;
        double total_gpu_time = 0;
        long long total_ops = 0;
        int pairs = 0;
        
        for (int run = 0; run < NUM_RUNS; run++) {
            auto positions = generateDiagonalPositions(N, pins, 20);
            
            double cpu_time, gpu_time;
            auto cpu_results = cpu_finder.findOptimalPaths(positions, cost_grid, cpu_time);
            gpu_finder.findOptimalPaths(positions, cost_grid, gpu_time);
            
            total_cpu_time += cpu_time;
            total_gpu_time += gpu_time;
            
            if (run == 0) {
                pairs = cpu_results.size();
                for (const auto& res : cpu_results) {
                    if (res.valid) {
                        total_ops += res.square_size; // One op per diagonal point
                    }
                }
            }
        }
        
        double avg_cpu_time = total_cpu_time / NUM_RUNS / 1000.0; // Convert to seconds
        double avg_gpu_time = total_gpu_time / NUM_RUNS / 1000.0;
        
        double cpu_mops = (avg_cpu_time > 0) ? (total_ops / avg_cpu_time / 1e6) : 0;
        double gpu_mops = (avg_gpu_time > 0) ? (total_ops / avg_gpu_time / 1e6) : 0;
        double ratio = (cpu_mops > 0) ? gpu_mops / cpu_mops : 0;
        
        cout << fixed << setprecision(1);
        cout << setw(12) << pins
             << setw(12) << pairs
             << setw(15) << (pairs > 0 ? total_ops / pairs : 0)
             << setw(15) << total_ops
             << setw(15) << cpu_mops
             << setw(15) << gpu_mops
             << setw(15) << ratio << endl;
        
        if (pins >= 50000) break; // Don't test larger for time reasons
    }
}

// ==================== MEMORY SCALING ANALYSIS ====================
void analyzeMemoryScaling() {
    cout << "\n\n==========================================" << endl;
    cout << "MEMORY SCALING ANALYSIS" << endl;
    cout << "==========================================" << endl;
    
    vector<int> grid_sizes = {100, 500, 1000, 2000};
    vector<int> pin_counts = {100, 1000, 10000};
    
    cout << "\nMemory requirements for different configurations:" << endl;
    cout << "(Assuming 4 bytes per int/coordinate)" << endl;
    cout << "\n" << string(90, '=') << endl;
    cout << setw(12) << "Grid Size"
         << setw(12) << "Pins"
         << setw(20) << "Grid Memory (MB)"
         << setw(20) << "Points Memory (KB)"
         << setw(20) << "Results Memory (KB)" << endl;
    cout << string(90, '=') << endl;
    
    for (int N : grid_sizes) {
        for (int pins : pin_counts) {
            size_t grid_memory = sizeof(int) * N * N;
            size_t points_memory = sizeof(Point) * pins;
            size_t results_memory = sizeof(PathResult) * (pins - 1);
            
            cout << setw(12) << N << "x" << N
                 << setw(12) << pins
                 << setw(20) << fixed << setprecision(1) << grid_memory / (1024.0*1024.0)
                 << setw(20) << fixed << setprecision(1) << points_memory / 1024.0
                 << setw(20) << fixed << setprecision(1) << results_memory / 1024.0 << endl;
        }
    }
    
    cout << "\nGPU Memory Considerations:" << endl;
    cout << "1. Cost grid: Largest memory consumer (N² × 4 bytes)" << endl;
    cout << "2. For N=2000: Grid alone requires 16MB" << endl;
    cout << "3. Points and results are negligible in comparison" << endl;
    cout << "4. Memory transfer time dominates for small pin counts" << endl;
}

// ==================== MAIN ====================
int main() {
    srand(42); // For reproducible results
    
    cout << "==========================================" << endl;
    cout << "COMPREHENSIVE PERFORMANCE ANALYSIS" << endl;
    cout << "GPU vs CPU: Minimum Cost Path Finder" << endl;
    cout << "==========================================" << endl;
    
    // Initialize CUDA
    cudaError_t cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaSetDevice failed! Do you have a CUDA-capable GPU?" << endl;
        return 1;
    }
    
    try {
        // Run all analyses
        runPinCountComparison();
        analyzeBreakEvenPoint();
        analyzeThroughput();
        analyzeMemoryScaling();
        
        // Print GPU info
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        
        cout << "\n\n==========================================" << endl;
        cout << "SYSTEM INFORMATION" << endl;
        cout << "==========================================" << endl;
        cout << "GPU: " << prop.name << endl;
        cout << "Compute Capability: " << prop.major << "." << prop.minor << endl;
        cout << "Global Memory: " << prop.totalGlobalMem / (1024*1024*1024) << " GB" << endl;
        cout << "Multiprocessors: " << prop.multiProcessorCount << endl;
        cout << "Max Threads per Block: " << prop.maxThreadsPerBlock << endl;
        cout << "Clock Rate: " << prop.clockRate / 1000 << " MHz" << endl;
        
    } catch (const exception& e) {
        cerr << "\nERROR: " << e.what() << endl;
        return 1;
    } catch (...) {
        cerr << "\nUNKNOWN ERROR" << endl;
        return 1;
    }
    
    cudaDeviceReset();
    
    cout << "\n==========================================" << endl;
    cout << "ANALYSIS COMPLETED SUCCESSFULLY" << endl;
    cout << "==========================================" << endl;
    
    return 0;
}