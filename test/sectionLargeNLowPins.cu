#include <vector>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cstdlib>
#include <cuda_runtime.h>

using namespace std;
using namespace std::chrono;

// ==================== DATA STRUCTURES ====================
struct Point {
    int x, y;
};

struct Rectangle {
    Point top_left;
    Point bottom_right;
    int width;
    int height;
    int type;
};

// ==================== OPTIMIZED GPU KERNELS ====================
__global__ void extractRectanglesKernel(
    const Point* points,
    Rectangle* rectangles,
    int* rect_counts,
    int num_points,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_points - 1) {
        Point p1 = points[idx];
        Point p2 = points[idx + 1];
        
        if (p1.x <= p2.x && p1.y <= p2.y) {
            int dr = p2.x - p1.x;
            int dc = p2.y - p1.y;
            
            int n = max(dr, dc) + 1;
            n = min(n, N - p1.x);
            n = min(n, N - p1.y);
            n = max(n, 1);
            
            Point tl = p1;
            Point br = {tl.x + n - 1, tl.y + n - 1};
            
            // Generate Type A-C rectangles (TL -> diagonal)
            for (int i = 0; i < n; i++) {
                int rect_idx = idx * 2 * 100 + i; // Safe limit: 100 max square size
                Point point_on_diag = {tl.x + i, tl.y + (n - 1 - i)};
                rectangles[rect_idx] = {
                    tl, point_on_diag, 
                    point_on_diag.x - tl.x + 1,
                    point_on_diag.y - tl.y + 1,
                    0
                };
            }
            
            // Generate Type B-D rectangles (diagonal -> BR)
            for (int i = 0; i < n; i++) {
                int rect_idx = idx * 2 * 100 + 100 + i; // Offset by max square size
                Point point_on_diag = {tl.x + i, tl.y + (n - 1 - i)};
                rectangles[rect_idx] = {
                    point_on_diag, br,
                    br.x - point_on_diag.x + 1,
                    br.y - point_on_diag.y + 1,
                    1
                };
            }
            
            rect_counts[idx] = 2 * n;
        } else {
            rect_counts[idx] = 0;
        }
    }
}

// ==================== SAFE GPU CLASS ====================
class SafeGPUExtractor {
private:
    Point* d_points;
    Rectangle* d_rectangles;
    int* d_rect_counts;
    int max_points;
    int max_rect_per_square;
    
public:
    SafeGPUExtractor(int max_points_count = 10000, int max_square_size = 100) 
        : max_points(max_points_count), max_rect_per_square(2 * max_square_size) {
        
        size_t points_mem = sizeof(Point) * max_points_count;
        size_t counts_mem = sizeof(int) * max_points_count;
        size_t rects_mem = sizeof(Rectangle) * max_points_count * max_rect_per_square;
        
        cout << "Allocating GPU memory:" << endl;
        cout << "  Points: " << points_mem / 1024 << " KB" << endl;
        cout << "  Counts: " << counts_mem / 1024 << " KB" << endl;
        cout << "  Rectangles: " << rects_mem / (1024*1024) << " MB" << endl;
        
        cudaError_t err;
        err = cudaMalloc(&d_points, points_mem);
        if (err != cudaSuccess) {
            cerr << "Failed to allocate points memory: " << cudaGetErrorString(err) << endl;
            exit(1);
        }
        
        err = cudaMalloc(&d_rect_counts, counts_mem);
        if (err != cudaSuccess) {
            cerr << "Failed to allocate counts memory: " << cudaGetErrorString(err) << endl;
            cudaFree(d_points);
            exit(1);
        }
        
        err = cudaMalloc(&d_rectangles, rects_mem);
        if (err != cudaSuccess) {
            cerr << "Failed to allocate rectangles memory: " << cudaGetErrorString(err) << endl;
            cudaFree(d_points);
            cudaFree(d_rect_counts);
            exit(1);
        }
    }
    
    ~SafeGPUExtractor() {
        cudaFree(d_points);
        cudaFree(d_rectangles);
        cudaFree(d_rect_counts);
    }
    
    pair<vector<vector<Rectangle>>, double> extractRectangles(
        const vector<pair<int, int>>& positions,
        int N
    ) {
        auto start = high_resolution_clock::now();
        
        int num_points = positions.size();
        if (num_points < 2) return {{}, 0};
        
        // Check if we can handle this square size
        int max_possible_square = 0;
        for (int i = 0; i < num_points - 1; i++) {
            Point p1 = {positions[i].first, positions[i].second};
            Point p2 = {positions[i+1].first, positions[i+1].second};
            if (p1.x <= p2.x && p1.y <= p2.y) {
                int dr = p2.x - p1.x;
                int dc = p2.y - p1.y;
                int n = max(dr, dc) + 1;
                if (n > max_possible_square) max_possible_square = n;
            }
        }
        
        if (max_possible_square * 2 > max_rect_per_square) {
            cerr << "Warning: Square size " << max_possible_square 
                 << " exceeds maximum supported size " << (max_rect_per_square / 2) << endl;
            return {{}, 0};
        }
        
        // Convert and copy points
        vector<Point> h_points(num_points);
        for (int i = 0; i < num_points; i++) {
            h_points[i] = {positions[i].first, positions[i].second};
        }
        
        cudaMemcpy(d_points, h_points.data(), 
                  sizeof(Point) * num_points, cudaMemcpyHostToDevice);
        
        // Launch kernel
        int block_size = 256;
        int num_blocks = (num_points + block_size - 1) / block_size;
        
        extractRectanglesKernel<<<num_blocks, block_size>>>(
            d_points, d_rectangles, d_rect_counts, num_points, N
        );
        
        cudaError_t kernel_err = cudaGetLastError();
        if (kernel_err != cudaSuccess) {
            cerr << "Kernel error: " << cudaGetErrorString(kernel_err) << endl;
            return {{}, 0};
        }
        
        cudaDeviceSynchronize();
        
        // Get rectangle counts
        vector<int> h_rect_counts(num_points - 1);
        cudaMemcpy(h_rect_counts.data(), d_rect_counts,
                  sizeof(int) * (num_points - 1), cudaMemcpyDeviceToHost);
        
        // Calculate total rectangles
        int total_rectangles = 0;
        int valid_squares = 0;
        for (int count : h_rect_counts) {
            if (count > 0) {
                total_rectangles += count;
                valid_squares++;
            }
        }
        
        // Get rectangles
        vector<vector<Rectangle>> results;
        if (total_rectangles > 0) {
            // Allocate only what we need
            vector<Rectangle> h_all_rectangles(total_rectangles);
            
            // Copy rectangles
            int current_idx = 0;
            for (int i = 0; i < num_points - 1; i++) {
                if (h_rect_counts[i] > 0) {
                    cudaMemcpy(&h_all_rectangles[current_idx], 
                              &d_rectangles[i * max_rect_per_square],
                              sizeof(Rectangle) * h_rect_counts[i],
                              cudaMemcpyDeviceToHost);
                    current_idx += h_rect_counts[i];
                }
            }
            
            // Organize by square
            results.resize(valid_squares);
            int square_idx = 0;
            current_idx = 0;
            
            for (int i = 0; i < num_points - 1; i++) {
                if (h_rect_counts[i] > 0) {
                    for (int j = 0; j < h_rect_counts[i]; j++) {
                        results[square_idx].push_back(h_all_rectangles[current_idx + j]);
                    }
                    current_idx += h_rect_counts[i];
                    square_idx++;
                }
            }
        }
        
        auto end = high_resolution_clock::now();
        double time_ms = duration_cast<microseconds>(end - start).count() / 1000.0;
        
        return {results, time_ms};
    }
};

// ==================== SAFE CPU CLASS ====================
class SafeCPUExtractor {
public:
    vector<vector<Rectangle>> extractRectangles(
        const vector<pair<int, int>>& positions,
        int N,
        double& time_ms
    ) {
        auto start = high_resolution_clock::now();
        
        vector<vector<Rectangle>> results;
        
        for (size_t i = 0; i < positions.size() - 1; i++) {
            Point p1 = {positions[i].first, positions[i].second};
            Point p2 = {positions[i+1].first, positions[i+1].second};
            
            if (p1.x <= p2.x && p1.y <= p2.y) {
                int dr = p2.x - p1.x;
                int dc = p2.y - p1.y;
                
                int n = max(dr, dc) + 1;
                n = min(n, N - p1.x);
                n = min(n, N - p1.y);
                n = max(n, 1);
                
                // Safety check: don't create huge squares
                if (n > 10000) {
                    cerr << "Warning: Skipping very large square of size " << n << endl;
                    continue;
                }
                
                Point tl = p1;
                Point br = {tl.x + n - 1, tl.y + n - 1};
                
                vector<Rectangle> square_rects;
                square_rects.reserve(2 * n);
                
                // Type A-C
                for (int j = 0; j < n; j++) {
                    Point point_on_diag = {tl.x + j, tl.y + (n - 1 - j)};
                    square_rects.push_back({
                        tl, point_on_diag,
                        point_on_diag.x - tl.x + 1,
                        point_on_diag.y - tl.y + 1,
                        0
                    });
                }
                
                // Type B-D
                for (int j = 0; j < n; j++) {
                    Point point_on_diag = {tl.x + j, tl.y + (n - 1 - j)};
                    square_rects.push_back({
                        point_on_diag, br,
                        br.x - point_on_diag.x + 1,
                        br.y - point_on_diag.y + 1,
                        1
                    });
                }
                
                results.push_back(square_rects);
            }
        }
        
        auto end = high_resolution_clock::now();
        time_ms = duration_cast<microseconds>(end - start).count() / 1000.0;
        
        return results;
    }
};

// ==================== PERFORMANCE TEST - SAFE VERSION ====================
void runSafePerformanceTest() {
    cout << fixed << setprecision(3);
    
    cout << "==========================================" << endl;
    cout << "SAFE PERFORMANCE TEST: LARGE GRID, SMALL POINTS" << endl;
    cout << "==========================================" << endl;
    cout << "Maximum square size limited to 100 for memory safety" << endl;
    cout << "==========================================" << endl << endl;
    
    // Test configurations: keep square sizes reasonable
    vector<pair<int, int>> test_configs = {
        {100, 10},    // Small grid, few points
        {100, 100},   // Small grid, more points
        {1000, 10},   // Medium grid, few points
        {1000, 100},  // Medium grid, more points
        {5000, 10},   // Large grid, few points
        {5000, 50},   // Large grid, moderate points
    };
    
    // Initialize with safe limits
    SafeGPUExtractor gpu_extractor(100000, 100); // Max 100K points, max square size 100
    SafeCPUExtractor cpu_extractor;
    
    cout << setw(12) << "Grid Size" 
         << setw(12) << "Points"
         << setw(15) << "Squares"
         << setw(15) << "Rectangles"
         << setw(15) << "CPU (ms)"
         << setw(15) << "GPU (ms)"
         << setw(15) << "CPU Faster"
         << setw(20) << "Overhead Ratio" << endl;
    cout << string(120, '=') << endl;
    
    for (const auto& [N, num_points] : test_configs) {
        // Generate positions with controlled square sizes
        vector<pair<int, int>> positions;
        int x = 0, y = 0;
        
        for (int i = 0; i < num_points; i++) {
            positions.push_back({x, y});
            
            // Control square size by controlling point spacing
            int max_step = min(50, N - x - 10); // Limit square size to ~50
            max_step = max(max_step, 1);
            
            int step_x = 1 + rand() % max_step;
            int step_y = 1 + rand() % max_step;
            
            x = min(N - 1, x + step_x);
            y = min(N - 1, y + step_y);
        }
        
        try {
            // CPU execution
            double cpu_time;
            auto cpu_results = cpu_extractor.extractRectangles(positions, N, cpu_time);
            
            // GPU execution
            auto [gpu_results, gpu_time] = gpu_extractor.extractRectangles(positions, N);
            
            // Calculate statistics
            int valid_squares = cpu_results.size();
            int total_rectangles = 0;
            for (const auto& rects : cpu_results) {
                total_rectangles += rects.size();
            }
            
            double speedup = cpu_time / gpu_time;
            double overhead_ratio = (gpu_time > cpu_time) ? 
                                   (gpu_time - cpu_time) / cpu_time * 100 : 
                                   (cpu_time - gpu_time) / gpu_time * 100;
            
            cout << setw(12) << N << "x" << N
                 << setw(12) << num_points
                 << setw(15) << valid_squares
                 << setw(15) << total_rectangles
                 << setw(15) << cpu_time
                 << setw(15) << gpu_time
                 << setw(15) << (cpu_time < gpu_time ? "Yes" : "No")
                 << setw(20) << overhead_ratio << "%" << endl;
            
            // Simple validation
            if (!cpu_results.empty() && !gpu_results.empty()) {
                if (cpu_results.size() != gpu_results.size()) {
                    cout << "  WARNING: Square count mismatch!" << endl;
                }
            }
            
        } catch (const exception& e) {
            cout << "  ERROR for grid " << N << "x" << N << " with " << num_points 
                 << " points: " << e.what() << endl;
        } catch (...) {
            cout << "  UNKNOWN ERROR for grid " << N << "x" << N 
                 << " with " << num_points << " points" << endl;
        }
    }
}

// ==================== MEMORY ANALYSIS ====================
void analyzeMemoryRequirements() {
    cout << "\n\n==========================================" << endl;
    cout << "MEMORY REQUIREMENT ANALYSIS" << endl;
    cout << "==========================================" << endl;
    
    cout << "\nFor N by N grid with M points:" << endl;
    cout << "------------------------------------------" << endl;
    
    // CPU memory
    cout << "CPU Memory:" << endl;
    cout << "  - Points storage: " << sizeof(Point) << " bytes per point" << endl;
    cout << "  - Rectangle storage: " << sizeof(Rectangle) << " bytes per rectangle" << endl;
    cout << "  - Max rectangles per square: 2*N" << endl;
    cout << "  - Total worst case: M * 2 * N * " << sizeof(Rectangle) << " bytes" << endl;
    
    cout << "\nExample calculations:" << endl;
    cout << "  N=1000, M=100: " << (100 * 2 * 1000 * sizeof(Rectangle) / (1024*1024)) << " MB" << endl;
    cout << "  N=10000, M=100: " << (100 * 2 * 10000 * sizeof(Rectangle) / (1024*1024)) << " MB" << endl;
    cout << "  N=100000, M=100: " << (100 * 2 * 100000 * sizeof(Rectangle) / (1024*1024)) << " MB" << endl;
}

// ==================== MAIN - SAFE VERSION ====================
int main() {
    srand(42);
    
    cout << "==========================================" << endl;
    cout << "SAFE GPU vs CPU ANALYSIS" << endl;
    cout << "Memory-safe version with limits" << endl;
    cout << "==========================================" << endl << endl;
    
    try {
        runSafePerformanceTest();
        analyzeMemoryRequirements();
    } catch (const exception& e) {
        cerr << "\nERROR: " << e.what() << endl;
        return 1;
    } catch (...) {
        cerr << "\nUNKNOWN ERROR" << endl;
        return 1;
    }
    
    cudaDeviceReset();
    
    cout << "\n==========================================" << endl;
    cout << "TEST COMPLETED SUCCESSFULLY" << endl;
    cout << "==========================================" << endl;
    
    return 0;
}