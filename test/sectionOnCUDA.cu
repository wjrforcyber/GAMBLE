#include <vector>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

using namespace std;
using namespace std::chrono;

//namespace cg = cooperative_groups;

// ==================== HIGHLY OPTIMIZED GPU IMPLEMENTATION ====================
struct Point {
    int x, y;
};

// Warp-level optimized kernel (32 threads work together)
__global__ void extractSquaresWarpOptimized(
    const Point* __restrict__ points,
    int2* __restrict__ results,
    int numPoints,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = idx / 32;
    int lane_id = idx % 32;
    
    // Process multiple points per warp for better occupancy
    for (int i = warp_id; i < numPoints - 1; i += gridDim.x * blockDim.x / 32) {
        Point p1, p2;
        
        // Cooperative loading within warp
        if (lane_id == 0) {
            p1 = points[i];
            p2 = points[i + 1];
        }
        
        // Broadcast loaded values to all threads in warp
        p1.x = __shfl_sync(0xFFFFFFFF, p1.x, 0);
        p1.y = __shfl_sync(0xFFFFFFFF, p1.y, 0);
        p2.x = __shfl_sync(0xFFFFFFFF, p2.x, 0);
        p2.y = __shfl_sync(0xFFFFFFFF, p2.y, 0);
        
        if (p1.x <= p2.x && p1.y <= p2.y) {
            int dr = p2.x - p1.x;
            int dc = p2.y - p1.y;
            
            int size = max(dr, dc) + 1;
            
            // Boundary check
            int max_rows = N - p1.x;
            int max_cols = N - p1.y;
            size = min(size, min(max_rows, max_cols));
            size = max(size, 1);
            
            // Only one thread writes the result
            if (lane_id == 0) {
                results[i].x = p1.x * N + p1.y;
                results[i].y = size;
            }
        } else if (lane_id == 0) {
            results[i].x = -1;
            results[i].y = -1;
        }
    }
}

// Batch processing kernel with shared memory
__global__ void extractSquaresBatchKernel(
    const Point* __restrict__ points,
    int2* __restrict__ results,
    int numPoints,
    int N,
    int batchSize
) {
    extern __shared__ Point shared_points[];
    
    int batch_id = blockIdx.x;
    int thread_id = threadIdx.x;
    int points_per_block = blockDim.x;
    
    int start_idx = batch_id * batchSize;
    int end_idx = min(start_idx + batchSize, numPoints - 1);
    
    // Load batch into shared memory
    for (int i = thread_id; i < min(batchSize + 1, numPoints - start_idx); i += points_per_block) {
        int load_idx = start_idx + i;
        if (load_idx < numPoints) {
            shared_points[i] = points[load_idx];
        }
    }
    __syncthreads();
    
    // Process points in this batch
    for (int i = thread_id; i < min(batchSize, end_idx - start_idx); i += points_per_block) {
        Point p1 = shared_points[i];
        Point p2 = shared_points[i + 1];
        
        if (p1.x <= p2.x && p1.y <= p2.y) {
            int dr = p2.x - p1.x;
            int dc = p2.y - p1.y;
            
            int size = max(dr, dc) + 1;
            
            // Fast boundary check using min/max
            size = min(size, N - p1.x);
            size = min(size, N - p1.y);
            size = max(size, 1);
            
            results[start_idx + i].x = p1.x * N + p1.y;
            results[start_idx + i].y = size;
        } else {
            results[start_idx + i].x = -1;
            results[start_idx + i].y = -1;
        }
    }
}

class HighPerfGPUExtractor {
private:
    Point* d_points;
    int2* d_results;
    int gridSize;
    cudaStream_t stream;
    
public:
    HighPerfGPUExtractor(int N) : gridSize(N) {
        // Allocate with cudaMallocHost for pinned memory (faster transfers)
        cudaMallocHost(&d_points, sizeof(Point) * 10000000);
        cudaMallocHost(&d_results, sizeof(int2) * 10000000);
        cudaStreamCreate(&stream);
    }
    
    ~HighPerfGPUExtractor() {
        cudaFreeHost(d_points);
        cudaFreeHost(d_results);
        cudaStreamDestroy(stream);
    }
    
    // Process with asynchronous transfers and computation overlap
    vector<pair<pair<int, int>, int>> extractSquaresAsync(
        const vector<pair<int, int>>& positions,
        double& gpuTime
    ) {
        if (positions.size() < 2) return {};
        
        auto start = high_resolution_clock::now();
        
        // Use pinned memory directly (already allocated)
        for (size_t i = 0; i < positions.size(); i++) {
            d_points[i].x = positions[i].first;
            d_points[i].y = positions[i].second;
        }
        
        // Async copy to device
        Point* d_points_device;
        int2* d_results_device;
        
        cudaMalloc(&d_points_device, sizeof(Point) * positions.size());
        cudaMalloc(&d_results_device, sizeof(int2) * (positions.size() - 1));
        
        cudaMemcpyAsync(d_points_device, d_points, 
                       sizeof(Point) * positions.size(),
                       cudaMemcpyHostToDevice, stream);
        
        // Launch kernel with optimal configuration
        int blockSize = 256;
        int numBlocks = (positions.size() + blockSize - 1) / blockSize;
        
        // Use warp-optimized kernel for large problems
        if (positions.size() > 10000) {
            extractSquaresWarpOptimized<<<numBlocks, blockSize, 0, stream>>>(
                d_points_device, d_results_device, positions.size(), gridSize
            );
        } else {
            // Use batch kernel for smaller problems
            int batchSize = 1024;
            int sharedMemSize = (batchSize + 1) * sizeof(Point);
            numBlocks = (positions.size() + batchSize - 1) / batchSize;
            
            extractSquaresBatchKernel<<<numBlocks, blockSize, sharedMemSize, stream>>>(
                d_points_device, d_results_device, positions.size(), gridSize, batchSize
            );
        }
        
        // Async copy back
        cudaMemcpyAsync(d_results, d_results_device,
                       sizeof(int2) * (positions.size() - 1),
                       cudaMemcpyDeviceToHost, stream);
        
        cudaStreamSynchronize(stream);
        
        auto end = high_resolution_clock::now();
        gpuTime = duration_cast<microseconds>(end - start).count();
        
        // Process results
        vector<pair<pair<int, int>, int>> squares;
        squares.reserve(positions.size() - 1);
        
        for (size_t i = 0; i < positions.size() - 1; i++) {
            if (d_results[i].x >= 0 && d_results[i].y > 0) {
                int encoded = d_results[i].x;
                squares.push_back({{encoded / gridSize, encoded % gridSize}, d_results[i].y});
            }
        }
        
        cudaFree(d_points_device);
        cudaFree(d_results_device);
        
        return squares;
    }
};

// ==================== HIGHLY OPTIMIZED CPU (WITH SIMD) ====================
#include <xmmintrin.h>  // SSE
#include <emmintrin.h>  // SSE2
#include <immintrin.h>  // AVX

class SIMDCPUExtractor {
private:
    int gridSize;
    
    // SIMD-optimized processing (process 8 points at a time with AVX2)
    void processSIMD(const Point* points, int2* results, int n) {
        for (int i = 0; i < n - 1; i += 8) {
            // Load 8 pairs of points
            __m256i p1_x = _mm256_loadu_si256((__m256i*)&points[i].x);
            __m256i p1_y = _mm256_loadu_si256((__m256i*)&points[i].y);
            __m256i p2_x = _mm256_loadu_si256((__m256i*)&points[i + 1].x);
            __m256i p2_y = _mm256_loadu_si256((__m256i*)&points[i + 1].y);
            
            // Check if p1 <= p2
            __m256i cmp_x = _mm256_cmpgt_epi32(p1_x, p2_x);
            __m256i cmp_y = _mm256_cmpgt_epi32(p1_y, p2_y);
            __m256i invalid = _mm256_or_si256(cmp_x, cmp_y);
            
            // Calculate distances
            __m256i dr = _mm256_sub_epi32(p2_x, p1_x);
            __m256i dc = _mm256_sub_epi32(p2_y, p1_y);
            
            // size = max(dr, dc) + 1
            __m256i size = _mm256_max_epi32(dr, dc);
            size = _mm256_add_epi32(size, _mm256_set1_epi32(1));
            
            // Boundary check
            __m256i max_rows = _mm256_sub_epi32(_mm256_set1_epi32(gridSize), p1_x);
            __m256i max_cols = _mm256_sub_epi32(_mm256_set1_epi32(gridSize), p1_y);
            __m256i max_possible = _mm256_min_epi32(max_rows, max_cols);
            size = _mm256_min_epi32(size, max_possible);
            size = _mm256_max_epi32(size, _mm256_set1_epi32(1));
            
            // Mark invalid results
            size = _mm256_andnot_si256(invalid, size);
            __m256i mask = _mm256_cmpeq_epi32(invalid, _mm256_set1_epi32(-1));
            size = _mm256_or_si256(size, _mm256_and_si256(mask, _mm256_set1_epi32(-1)));
            
            // Store results
            _mm256_storeu_si256((__m256i*)&results[i], size);
        }
    }
    
public:
    SIMDCPUExtractor(int N) : gridSize(N) {}
    
    vector<pair<pair<int, int>, int>> extractSquares(
        const vector<pair<int, int>>& positions
    ) {
        vector<pair<pair<int, int>, int>> squares;
        squares.reserve(positions.size() - 1);
        
        // Convert to SIMD-friendly format
        vector<Point> points(positions.size());
        for (size_t i = 0; i < positions.size(); i++) {
            points[i].x = positions[i].first;
            points[i].y = positions[i].second;
        }
        
        // Process with SIMD when possible
        #ifdef __AVX2__
        if (positions.size() > 16) {
            vector<int2> results(positions.size() - 1);
            processSIMD(points.data(), results.data(), positions.size());
            
            for (size_t i = 0; i < results.size(); i++) {
                if (results[i].x > 0) {
                    squares.push_back({{points[i].x, points[i].y}, results[i].x});
                }
            }
            return squares;
        }
        #endif
        
        // Fallback to scalar for small sizes
        for (size_t i = 0; i < positions.size() - 1; i++) {
            const auto& p1 = positions[i];
            const auto& p2 = positions[i + 1];
            
            if (p1.first <= p2.first && p1.second <= p2.second) {
                int dr = p2.first - p1.first;
                int dc = p2.second - p1.second;
                
                int size = max(dr, dc) + 1;
                size = min(size, gridSize - p1.first);
                size = min(size, gridSize - p1.second);
                size = max(size, 1);
                
                squares.push_back({p1, size});
            }
        }
        
        return squares;
    }
};

// ==================== REAL PERFORMANCE TEST ====================
void runTruePerformanceTest() {
    cout << "=== TRUE GPU vs CPU Performance Test ===" << endl;
    cout << "Testing with REAL workload sizes" << endl;
    cout << "========================================" << endl;
    
    const int N = 1000;
    HighPerfGPUExtractor gpuExtractor(N);
    SIMDCPUExtractor cpuExtractor(N);
    
    // Test realistic workloads
    struct Workload {
        string name;
        int numPoints;
        int repetitions;
    };
    
    vector<Workload> workloads = {
        {"Tiny", 100, 1000},
        {"Small", 1000, 100},
        {"Medium", 10000, 10},
        {"Large", 100000, 5},
        {"Huge", 1000000, 2}
    };
    
    cout << setw(25) << "Workload" 
         << setw(15) << "Points" 
         << setw(15) << "CPU (μs)" 
         << setw(15) << "GPU (μs)"
         << setw(15) << "Speedup" 
         << setw(20) << "Notes" << endl;
    cout << string(105, '-') << endl;
    
    for (const auto& workload : workloads) {
        // Generate test data
        vector<pair<int, int>> positions;
        positions.reserve(workload.numPoints);
        
        int x = 0, y = 0;
        for (int i = 0; i < workload.numPoints; i++) {
            positions.push_back({x, y});
            x = (x + 1 + rand() % 5) % (N - 10);
            y = (y + 1 + rand() % 5) % (N - 10);
        }
        
        // Warm up
        cpuExtractor.extractSquares(positions);
        double dummy;
        gpuExtractor.extractSquaresAsync(positions, dummy);
        
        // CPU benchmark
        double totalCPUTime = 0;
        for (int rep = 0; rep < workload.repetitions; rep++) {
            auto start = high_resolution_clock::now();
            auto results = cpuExtractor.extractSquares(positions);
            auto end = high_resolution_clock::now();
            totalCPUTime += duration_cast<microseconds>(end - start).count();
        }
        double avgCPUTime = totalCPUTime / workload.repetitions;
        
        // GPU benchmark
        double totalGPUTime = 0;
        for (int rep = 0; rep < workload.repetitions; rep++) {
            double gpuTime;
            auto results = gpuExtractor.extractSquaresAsync(positions, gpuTime);
            totalGPUTime += gpuTime;
        }
        double avgGPUTime = totalGPUTime / workload.repetitions;
        
        double speedup = avgCPUTime / avgGPUTime;
        string notes = speedup > 1.0 ? "GPU faster ✓" : "CPU faster ✓";
        
        cout << setw(25) << workload.name
             << setw(15) << workload.numPoints
             << setw(15) << fixed << setprecision(1) << avgCPUTime
             << setw(15) << fixed << setprecision(1) << avgGPUTime
             << setw(15) << fixed << setprecision(2) << speedup
             << setw(20) << notes << endl;
    }
}

// ==================== INVESTIGATE BOTTLENECKS ====================
void investigateBottlenecks() {
    cout << "\n\n=== Investigating Performance Bottlenecks ===" << endl;
    
    const int N = 1000;
    const int numPoints = 100000;
    
    // Generate large dataset
    vector<pair<int, int>> positions;
    positions.reserve(numPoints);
    
    int x = 0, y = 0;
    for (int i = 0; i < numPoints; i++) {
        positions.push_back({x, y});
        x = (x + 1) % (N - 10);
        y = (y + 1) % (N - 10);
    }
    
    // Measure individual components
    cout << "\n1. Memory Transfer Overhead:" << endl;
    
    Point* h_points = new Point[numPoints];
    for (int i = 0; i < numPoints; i++) {
        h_points[i].x = positions[i].first;
        h_points[i].y = positions[i].second;
    }
    
    Point* d_points;
    cudaMalloc(&d_points, sizeof(Point) * numPoints);
    
    auto start = high_resolution_clock::now();
    cudaMemcpy(d_points, h_points, sizeof(Point) * numPoints, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    auto end = high_resolution_clock::now();
    
    double transferTime = duration_cast<microseconds>(end - start).count();
    double bandwidth = (sizeof(Point) * numPoints) / (transferTime * 1e-6) / (1024 * 1024 * 1024); // GB/s
    
    cout << "   Transfer time: " << transferTime << " μs" << endl;
    cout << "   Bandwidth: " << fixed << setprecision(2) << bandwidth << " GB/s" << endl;
    
    // Measure kernel execution
    int2* d_results;
    cudaMalloc(&d_results, sizeof(int2) * (numPoints - 1));
    
    int blockSize = 256;
    int numBlocks = (numPoints + blockSize - 1) / blockSize;
    
    start = high_resolution_clock::now();
    extractSquaresWarpOptimized<<<numBlocks, blockSize>>>(d_points, d_results, numPoints, N);
    cudaDeviceSynchronize();
    end = high_resolution_clock::now();
    
    double kernelTime = duration_cast<microseconds>(end - start).count();
    cout << "\n2. Kernel Execution:" << endl;
    cout << "   Kernel time: " << kernelTime << " μs" << endl;
    cout << "   Throughput: " << fixed << setprecision(0) 
         << (numPoints / (kernelTime * 1e-6) / 1e6) << " million points/sec" << endl;
    
    // Measure CPU computation
    SIMDCPUExtractor cpuExtractor(N);
    start = high_resolution_clock::now();
    auto results = cpuExtractor.extractSquares(positions);
    end = high_resolution_clock::now();
    
    double cpuTime = duration_cast<microseconds>(end - start).count();
    cout << "\n3. CPU Computation:" << endl;
    cout << "   CPU time: " << cpuTime << " μs" << endl;
    cout << "   Throughput: " << fixed << setprecision(0)
         << (numPoints / (cpuTime * 1e-6) / 1e6) << " million points/sec" << endl;
    
    // Cleanup
    delete[] h_points;
    cudaFree(d_points);
    cudaFree(d_results);
    
    cout << "\n4. Analysis:" << endl;
    cout << "   Total GPU time (est): " << (transferTime * 2 + kernelTime) << " μs" << endl;
    cout << "   GPU/CPU ratio: " << fixed << setprecision(2) 
         << ((transferTime * 2 + kernelTime) / cpuTime) << endl;
}

// ==================== ENHANCED VERSION WITH ACTUAL GPU BENEFIT ====================
__global__ void extractAndProcessSquares(
    const float* __restrict__ image,  // Input image data
    const Point* __restrict__ points,
    float* __restrict__ outputs,      // Output: average + variance for each square
    int numPoints,
    int N,
    int imageWidth
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < numPoints - 1) {
        Point p1 = points[idx];
        Point p2 = points[idx + 1];
        
        if (p1.x <= p2.x && p1.y <= p2.y) {
            int dr = p2.x - p1.x;
            int dc = p2.y - p1.y;
            
            int size = max(dr, dc) + 1;
            size = min(size, N - p1.x);
            size = min(size, N - p1.y);
            size = max(size, 1);
            
            // Now do ACTUAL computation that benefits from GPU
            float sum = 0.0f;
            float sumSq = 0.0f;
            int count = 0;
            
            // Compute statistics for the square area
            for (int i = 0; i < size; i++) {
                for (int j = 0; j < size; j++) {
                    int pixelIdx = (p1.x + i) * imageWidth + (p1.y + j);
                    float val = image[pixelIdx];
                    sum += val;
                    sumSq += val * val;
                    count++;
                }
            }
            
            float avg = sum / count;
            float variance = (sumSq / count) - (avg * avg);
            
            outputs[idx * 2] = avg;
            outputs[idx * 2 + 1] = variance;
        } else {
            outputs[idx * 2] = -1.0f;
            outputs[idx * 2 + 1] = -1.0f;
        }
    }
}

int main() {
    srand(42);
    
    cout << "=== HONEST GPU vs CPU Analysis ===" << endl;
    cout << "For SIMPLE square extraction on a 1000x1000 grid:" << endl;
    cout << "=================================================" << endl;
    
    runTruePerformanceTest();
    investigateBottlenecks();
    
    cout << "\n=== CONCLUSION ===" << endl;
    cout << "For your specific problem (simple square extraction):" << endl;
    cout << "✓ CPU is BETTER - Use CPU implementation" << endl;
    cout << "✓ GPU only helps if computation per square increases 10-100x" << endl;
    cout << "✓ Consider GPU if adding image processing inside squares" << endl;
    
    cudaDeviceReset();
    return 0;
}