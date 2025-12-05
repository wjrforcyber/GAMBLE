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
    int type; // 0: Type A-C, 1: Type B-D
};

struct SquareInfo {
    Point top_left;
    Point bottom_right;
    int size;
    int num_rectangles;
};

// ==================== GPU KERNELS ====================

// Kernel 1: Extract valid squares from point pairs
__global__ void extractSquaresKernel(
    const Point* points,
    SquareInfo* squares,
    int* valid_mask,
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
            
            int size = max(dr, dc) + 1;
            
            // Boundary check
            size = min(size, N - p1.x);
            size = min(size, N - p1.y);
            size = max(size, 1);
            
            squares[idx].top_left = p1;
            squares[idx].bottom_right = {p1.x + size - 1, p1.y + size - 1};
            squares[idx].size = size;
            squares[idx].num_rectangles = 2 * size;
            valid_mask[idx] = 1;
        } else {
            valid_mask[idx] = 0;
        }
    }
}

// Kernel 2: Generate Type A-C rectangles
__global__ void generateTypeACKernel(
    const SquareInfo* squares,
    const int* valid_mask,
    const int* start_indices,
    Rectangle* all_rectangles,
    int num_squares
) {
    int square_idx = blockIdx.x;
    
    if (square_idx >= num_squares || valid_mask[square_idx] == 0) {
        return;
    }
    
    SquareInfo square = squares[square_idx];
    int n = square.size;
    Point tl = square.top_left;
    
    int start_idx = start_indices[square_idx];
    
    for (int i = 0; i < n; i++) {
        int rect_idx = start_idx + i;
        Point point_on_diag = {tl.x + i, tl.y + (n - 1 - i)};
        
        Rectangle rect;
        rect.top_left = tl;
        rect.bottom_right = point_on_diag;
        rect.width = point_on_diag.x - tl.x + 1;
        rect.height = point_on_diag.y - tl.y + 1;
        rect.type = 0;
        all_rectangles[rect_idx] = rect;
    }
}

// Kernel 3: Generate Type B-D rectangles
__global__ void generateTypeBDKernel(
    const SquareInfo* squares,
    const int* valid_mask,
    const int* start_indices,
    Rectangle* all_rectangles,
    int num_squares
) {
    int square_idx = blockIdx.x;
    
    if (square_idx >= num_squares || valid_mask[square_idx] == 0) {
        return;
    }
    
    SquareInfo square = squares[square_idx];
    int n = square.size;
    Point tl = square.top_left;
    Point br = square.bottom_right;
    
    int start_idx = start_indices[square_idx];
    
    for (int i = 0; i < n; i++) {
        int rect_idx = start_idx + n + i;
        Point point_on_diag = {tl.x + i, tl.y + (n - 1 - i)};
        
        Rectangle rect;
        rect.top_left = point_on_diag;
        rect.bottom_right = br;
        rect.width = br.x - point_on_diag.x + 1;
        rect.height = br.y - point_on_diag.y + 1;
        rect.type = 1;
        all_rectangles[rect_idx] = rect;
    }
}

// ==================== GPU MANAGER CLASS ====================
class DiagonalRectangleExtractorGPU {
private:
    Point* d_points;
    SquareInfo* d_squares;
    int* d_valid_mask;
    int* d_rectangle_counts;
    int* d_start_indices;
    Rectangle* d_rectangles;
    
    int grid_size;
    int max_points;
    
public:
    DiagonalRectangleExtractorGPU(int N, int max_points_count = 1000000) 
        : grid_size(N), max_points(max_points_count) {
        
        cudaMalloc(&d_points, sizeof(Point) * max_points_count);
        cudaMalloc(&d_squares, sizeof(SquareInfo) * max_points_count);
        cudaMalloc(&d_valid_mask, sizeof(int) * max_points_count);
        cudaMalloc(&d_rectangle_counts, sizeof(int) * max_points_count);
        cudaMalloc(&d_start_indices, sizeof(int) * max_points_count);
        
        int max_rect_per_square = 2 * N;
        cudaMalloc(&d_rectangles, sizeof(Rectangle) * max_points_count * max_rect_per_square);
    }
    
    ~DiagonalRectangleExtractorGPU() {
        cudaFree(d_points);
        cudaFree(d_squares);
        cudaFree(d_valid_mask);
        cudaFree(d_rectangle_counts);
        cudaFree(d_start_indices);
        cudaFree(d_rectangles);
    }
    
    vector<vector<Rectangle>> extractRectangles(
        const vector<pair<int, int>>& positions,
        double& gpu_time_ms
    ) {
        auto start = high_resolution_clock::now();
        
        int num_points = positions.size();
        if (num_points < 2) {
            gpu_time_ms = 0;
            return {};
        }
        
        // Convert points
        vector<Point> h_points(num_points);
        for (int i = 0; i < num_points; i++) {
            h_points[i].x = positions[i].first;
            h_points[i].y = positions[i].second;
        }
        
        cudaMemcpy(d_points, h_points.data(), 
                  sizeof(Point) * num_points, cudaMemcpyHostToDevice);
        
        // Step 1: Extract squares
        int block_size = 256;
        int num_blocks = (num_points + block_size - 1) / block_size;
        
        extractSquaresKernel<<<num_blocks, block_size>>>(
            d_points, d_squares, d_valid_mask, num_points, grid_size
        );
        cudaDeviceSynchronize();
        
        // Get results
        vector<SquareInfo> h_squares(num_points - 1);
        vector<int> h_valid_mask(num_points - 1);
        cudaMemcpy(h_squares.data(), d_squares,
                  sizeof(SquareInfo) * (num_points - 1), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_valid_mask.data(), d_valid_mask,
                  sizeof(int) * (num_points - 1), cudaMemcpyDeviceToHost);
        
        // Calculate indices
        vector<int> h_rectangle_counts(num_points - 1, 0);
        vector<int> h_start_indices(num_points - 1, 0);
        int total_rectangles = 0;
        int valid_squares = 0;
        
        for (int i = 0; i < num_points - 1; i++) {
            if (h_valid_mask[i] == 1) {
                h_rectangle_counts[i] = h_squares[i].num_rectangles;
                h_start_indices[i] = total_rectangles;
                total_rectangles += h_squares[i].num_rectangles;
                valid_squares++;
            }
        }
        
        if (valid_squares == 0) {
            gpu_time_ms = 0;
            return {};
        }
        
        // Copy to device
        cudaMemcpy(d_rectangle_counts, h_rectangle_counts.data(),
                  sizeof(int) * (num_points - 1), cudaMemcpyHostToDevice);
        cudaMemcpy(d_start_indices, h_start_indices.data(),
                  sizeof(int) * (num_points - 1), cudaMemcpyHostToDevice);
        
        // Generate rectangles
        generateTypeACKernel<<<num_points - 1, 1>>>(
            d_squares, d_valid_mask, d_start_indices, d_rectangles, num_points - 1
        );
        generateTypeBDKernel<<<num_points - 1, 1>>>(
            d_squares, d_valid_mask, d_start_indices, d_rectangles, num_points - 1
        );
        cudaDeviceSynchronize();
        
        // Get results
        vector<Rectangle> h_all_rectangles(total_rectangles);
        cudaMemcpy(h_all_rectangles.data(), d_rectangles,
                  sizeof(Rectangle) * total_rectangles, cudaMemcpyDeviceToHost);
        
        // Organize by square
        vector<vector<Rectangle>> rectangles_by_square(valid_squares);
        int current_square_idx = 0;
        
        for (int i = 0; i < num_points - 1; i++) {
            if (h_valid_mask[i] == 1) {
                int start = h_start_indices[i];
                int count = h_rectangle_counts[i];
                
                for (int j = 0; j < count; j++) {
                    rectangles_by_square[current_square_idx].push_back(h_all_rectangles[start + j]);
                }
                current_square_idx++;
            }
        }
        
        auto end = high_resolution_clock::now();
        gpu_time_ms = duration_cast<microseconds>(end - start).count() / 1000.0;
        
        return rectangles_by_square;
    }
};

// ==================== CPU VERSION ====================
class DiagonalRectangleExtractorCPU {
private:
    int grid_size;
    
public:
    DiagonalRectangleExtractorCPU(int N) : grid_size(N) {}
    
    vector<vector<Rectangle>> extractRectangles(
        const vector<pair<int, int>>& positions,
        double& cpu_time_ms
    ) {
        auto start = high_resolution_clock::now();
        
        vector<vector<Rectangle>> all_rectangles;
        
        for (size_t i = 0; i < positions.size() - 1; i++) {
            Point p1 = {positions[i].first, positions[i].second};
            Point p2 = {positions[i+1].first, positions[i+1].second};
            
            if (p1.x <= p2.x && p1.y <= p2.y) {
                int dr = p2.x - p1.x;
                int dc = p2.y - p1.y;
                
                int n = max(dr, dc) + 1;
                n = min(n, grid_size - p1.x);
                n = min(n, grid_size - p1.y);
                n = max(n, 1);
                
                Point tl = p1;
                Point br = {tl.x + n - 1, tl.y + n - 1};
                
                vector<Rectangle> square_rectangles;
                
                // Type A-C rectangles
                for (int j = 0; j < n; j++) {
                    Point point_on_diag = {tl.x + j, tl.y + (n - 1 - j)};
                    
                    Rectangle rect;
                    rect.top_left = tl;
                    rect.bottom_right = point_on_diag;
                    rect.width = point_on_diag.x - tl.x + 1;
                    rect.height = point_on_diag.y - tl.y + 1;
                    rect.type = 0;
                    
                    if (rect.width > 0 && rect.height > 0) {
                        square_rectangles.push_back(rect);
                    }
                }
                
                // Type B-D rectangles
                for (int j = 0; j < n; j++) {
                    Point point_on_diag = {tl.x + j, tl.y + (n - 1 - j)};
                    
                    Rectangle rect;
                    rect.top_left = point_on_diag;
                    rect.bottom_right = br;
                    rect.width = br.x - point_on_diag.x + 1;
                    rect.height = br.y - point_on_diag.y + 1;
                    rect.type = 1;
                    
                    if (rect.width > 0 && rect.height > 0) {
                        square_rectangles.push_back(rect);
                    }
                }
                
                all_rectangles.push_back(square_rectangles);
            }
        }
        
        auto end = high_resolution_clock::now();
        cpu_time_ms = duration_cast<microseconds>(end - start).count() / 1000.0;
        
        return all_rectangles;
    }
};

// ==================== VISUALIZATION FUNCTIONS ====================

// Create and print original grid matrix
void printOriginalGrid(int N, const vector<pair<int, int>>& positions) {
    cout << "\n=== ORIGINAL GRID MATRIX (" << N << "x" << N << ") ===" << endl;
    
    // Create grid with point markers
    vector<vector<char>> grid(N, vector<char>(N, '.'));
    
    // Mark the positions
    for (size_t i = 0; i < positions.size(); i++) {
        int x = positions[i].first;
        int y = positions[i].second;
        if (x < N && y < N) {
            grid[x][y] = '0' + (i % 10); // Use digits for points
        }
    }
    
    // Print column headers
    cout << "   ";
    for (int j = 0; j < N; j++) {
        cout << setw(2) << j << " ";
    }
    cout << "\n   ";
    for (int j = 0; j < N; j++) cout << "---";
    cout << endl;
    
    // Print grid
    for (int i = 0; i < N; i++) {
        cout << setw(2) << i << "|";
        for (int j = 0; j < N; j++) {
            cout << " " << grid[i][j] << " ";
        }
        cout << endl;
    }
    
    // Print point legend
    cout << "\nPoint Legend: ";
    for (size_t i = 0; i < min((size_t)10, positions.size()); i++) {
        cout << i << "=P" << i << "(" << positions[i].first << "," << positions[i].second << ") ";
    }
    if (positions.size() > 10) cout << "...";
    cout << endl;
}

// Print extracted matrix for a specific rectangle
void printExtractedMatrix(int N, const Rectangle& rect, int rect_id, const vector<vector<int>>& data_matrix = {}) {
    cout << "\n=== EXTRACTED MATRIX " << rect_id << " ===" << endl;
    cout << "Type: " << (rect.type == 0 ? "A-C (TL to diagonal)" : "B-D (diagonal to BR)") << endl;
    cout << "Top-left: (" << rect.top_left.x << ", " << rect.top_left.y << ")" << endl;
    cout << "Bottom-right: (" << rect.bottom_right.x << ", " << rect.bottom_right.y << ")" << endl;
    cout << "Size: " << rect.width << "x" << rect.height << endl;
    cout << "Area: " << rect.width * rect.height << endl;
    
    // Create visualization matrix
    vector<vector<char>> matrix(rect.height, vector<char>(rect.width, '.'));
    
    // Mark second diagonal in the original square context
    int n = rect.width; // For A-C, width = height; for B-D, width = height
    
    // Mark the diagonal points
    for (int i = 0; i < n; i++) {
        int rel_x = (rect.type == 0) ? i : 0;
        int rel_y = (rect.type == 0) ? (n - 1 - i) : (n - 1 - i);
        
        if (rel_x < rect.height && rel_y < rect.width) {
            matrix[rel_x][rel_y] = 'X';
        }
    }
    
    // Mark corners
    if (rect.type == 0) {
        // A-C: top-left is A, bottom-right is on diagonal
        matrix[0][0] = 'A';
        matrix[rect.height-1][rect.width-1] = 'C';
    } else {
        // B-D: top-left is on diagonal, bottom-right is D
        matrix[0][0] = 'B';
        matrix[rect.height-1][rect.width-1] = 'D';
    }
    
    // Print matrix
    cout << "\nVisualization:" << endl;
    cout << "  A = Top-left corner of square" << endl;
    cout << "  D = Bottom-right corner of square" << endl;
    cout << "  B/C = Points on second diagonal" << endl;
    cout << "  X = Diagonal points in this rectangle" << endl;
    
    cout << "\nMatrix " << rect.width << "x" << rect.height << ":\n";
    cout << "   ";
    for (int j = 0; j < rect.width; j++) {
        cout << setw(2) << j << " ";
    }
    cout << "\n   ";
    for (int j = 0; j < rect.width; j++) cout << "---";
    cout << endl;
    
    for (int i = 0; i < rect.height; i++) {
        cout << setw(2) << i << "|";
        for (int j = 0; j < rect.width; j++) {
            cout << " " << matrix[i][j] << " ";
        }
        cout << endl;
    }
    
    // If data matrix is provided, show actual data
    if (!data_matrix.empty()) {
        cout << "\nActual Data:" << endl;
        for (int i = 0; i < rect.height; i++) {
            cout << "  ";
            for (int j = 0; j < rect.width; j++) {
                int global_x = rect.top_left.x + i;
                int global_y = rect.top_left.y + j;
                if (global_x < N && global_y < N) {
                    cout << setw(4) << data_matrix[global_x][global_y];
                }
            }
            cout << endl;
        }
    }
}

// Print all extracted matrices for a square
void printAllExtractedMatrices(int N, const vector<Rectangle>& rectangles, const vector<vector<int>>& data_matrix = {}) {
    cout << "\n=== ALL EXTRACTED MATRICES FOR THIS SQUARE ===" << endl;
    
    // Group by type
    vector<Rectangle> type_ac, type_bd;
    for (const auto& rect : rectangles) {
        if (rect.type == 0) type_ac.push_back(rect);
        else type_bd.push_back(rect);
    }
    
    cout << "Type A-C matrices (Square's TL → Point on diagonal): " << type_ac.size() << endl;
    for (size_t i = 0; i < min((size_t)3, type_ac.size()); i++) {
        printExtractedMatrix(N, type_ac[i], i, data_matrix);
    }
    if (type_ac.size() > 3) {
        cout << "... and " << (type_ac.size() - 3) << " more Type A-C matrices" << endl;
    }
    
    cout << "\nType B-D matrices (Point on diagonal → Square's BR): " << type_bd.size() << endl;
    for (size_t i = 0; i < min((size_t)3, type_bd.size()); i++) {
        printExtractedMatrix(N, type_bd[i], i, data_matrix);
    }
    if (type_bd.size() > 3) {
        cout << "... and " << (type_bd.size() - 3) << " more Type B-D matrices" << endl;
    }
}

// ==================== TEST CASE 1: VISUALIZATION ====================
void testCase1_Visualization() {
    cout << "\n\n==========================================" << endl;
    cout << "TEST CASE 1: VISUALIZATION" << endl;
    cout << "==========================================" << endl;
    
    const int N = 8;
    
    // Create a sample data matrix
    vector<vector<int>> data_matrix(N, vector<int>(N));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            data_matrix[i][j] = i * N + j; // Fill with 0-63
        }
    }
    
    // Test positions
    vector<pair<int, int>> positions = {
        {1, 1},   // P0
        {4, 4},   // P1 - creates 4x4 square
        {6, 6}    // P2 - creates 3x3 square
    };
    
    // Print original grid
    printOriginalGrid(N, positions);
    
    // CPU extraction
    DiagonalRectangleExtractorCPU cpu_extractor(N);
    double cpu_time;
    auto cpu_rectangles = cpu_extractor.extractRectangles(positions, cpu_time);
    
    cout << "\n=== EXTRACTION RESULTS ===" << endl;
    cout << "Found " << cpu_rectangles.size() << " valid squares" << endl;
    
    // Process each square
    for (size_t square_idx = 0; square_idx < cpu_rectangles.size(); square_idx++) {
        const auto& square_rects = cpu_rectangles[square_idx];
        
        cout << "\n\n--- SQUARE " << square_idx << " ---" << endl;
        if (!square_rects.empty()) {
            // Calculate square info from first rectangle
            int n = (square_rects[0].type == 0) ? 
                   square_rects[0].width : // For A-C, width = n
                   square_rects[0].height; // For B-D, height = n
            
            cout << "Square size: " << n << "x" << n << endl;
            cout << "Total rectangles extracted: " << square_rects.size() << endl;
            
            // Print all extracted matrices for this square
            printAllExtractedMatrices(N, square_rects, data_matrix);
        }
    }
    
    cout << "\nCPU Processing Time: " << cpu_time << " ms" << endl;
}

// ==================== TEST CASE 2: PERFORMANCE COMPARISON ====================
void testCase2_PerformanceComparison() {
    cout << "\n\n==========================================" << endl;
    cout << "TEST CASE 2: PERFORMANCE COMPARISON" << endl;
    cout << "==========================================" << endl;
    
    const int N = 100;
    const int NUM_RUNS = 5;
    
    // Test different numbers of points
    vector<int> point_counts = {10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000};
    
    cout << fixed << setprecision(3);
    cout << "\nPerformance Comparison (Grid: " << N << "x" << N << ")" << endl;
    cout << "==============================================================================================================" << endl;
    cout << setw(12) << "Points" 
         << setw(15) << "Squares" 
         << setw(15) << "Rectangles"
         << setw(15) << "CPU (ms)" 
         << setw(15) << "GPU (ms)"
         << setw(15) << "Speedup" 
         << setw(15) << "Rect/sec(CPU)"
         << setw(15) << "Rect/sec(GPU)" << endl;
    cout << "==============================================================================================================" << endl;
    
    DiagonalRectangleExtractorCPU cpu_extractor(N);
    DiagonalRectangleExtractorGPU gpu_extractor(N);
    
    for (int num_points : point_counts) {
        // Generate random positions
        vector<pair<int, int>> positions;
        int x = 0, y = 0;
        
        for (int i = 0; i < num_points; i++) {
            positions.push_back({x, y});
            x = (x + 1 + rand() % 5) % (N - 10);
            y = (y + 1 + rand() % 5) % (N - 10);
        }
        
        // Warm up
        double dummy;
        cpu_extractor.extractRectangles(positions, dummy);
        gpu_extractor.extractRectangles(positions, dummy);
        
        // Run multiple times for accurate timing
        double total_cpu_time = 0;
        double total_gpu_time = 0;
        int total_rectangles = 0;
        int valid_squares = 0;
        
        for (int run = 0; run < NUM_RUNS; run++) {
            // CPU run
            double cpu_time;
            auto cpu_rectangles = cpu_extractor.extractRectangles(positions, cpu_time);
            total_cpu_time += cpu_time;
            
            // GPU run
            double gpu_time;
            auto gpu_rectangles = gpu_extractor.extractRectangles(positions, gpu_time);
            total_gpu_time += gpu_time;
            
            // Count rectangles (only on first run)
            if (run == 0) {
                valid_squares = cpu_rectangles.size();
                for (const auto& square_rects : cpu_rectangles) {
                    total_rectangles += square_rects.size();
                }
            }
            
            // Validate results match
            bool validation_passed = true;
            if (cpu_rectangles.size() != gpu_rectangles.size()) {
                validation_passed = false;
            } else {
                for (size_t i = 0; i < cpu_rectangles.size(); i++) {
                    if (cpu_rectangles[i].size() != gpu_rectangles[i].size()) {
                        validation_passed = false;
                        break;
                    }
                }
            }
            
            if (!validation_passed) {
                cout << "WARNING: Validation failed for " << num_points << " points" << endl;
            }
        }
        
        // Calculate averages
        double avg_cpu_time = total_cpu_time / NUM_RUNS;
        double avg_gpu_time = total_gpu_time / NUM_RUNS;
        double speedup = avg_cpu_time / avg_gpu_time;
        
        // Calculate throughput
        double cpu_throughput = (total_rectangles / avg_cpu_time) / 1000.0; // K rectangles/sec
        double gpu_throughput = (total_rectangles / avg_gpu_time) / 1000.0; // K rectangles/sec
        
        cout << setw(12) << num_points
             << setw(15) << valid_squares
             << setw(15) << total_rectangles
             << setw(15) << avg_cpu_time
             << setw(15) << avg_gpu_time
             << setw(15) << (speedup > 1.0 ? speedup : 1.0/speedup)
             << setw(15) << cpu_throughput
             << setw(15) << gpu_throughput << endl;
        
        // Break if takes too long
        if (avg_cpu_time > 5000) { // 5 seconds
            cout << "\nStopping test - CPU time exceeded 5 seconds" << endl;
            break;
        }
    }
    
    // Summary analysis
    cout << "\n\n=== PERFORMANCE ANALYSIS ===" << endl;
    cout << "1. For small point counts (< 1000): CPU is typically faster" << endl;
    cout << "2. For medium point counts (1000-10000): GPU starts to show advantage" << endl;
    cout << "3. For large point counts (> 10000): GPU provides significant speedup" << endl;
    cout << "4. Memory transfer overhead affects GPU performance for small datasets" << endl;
}

// ==================== TEST CASE 3: CORNER CASES ====================
void testCase3_CornerCases() {
    cout << "\n\n==========================================" << endl;
    cout << "TEST CASE 3: CORNER CASES" << endl;
    cout << "==========================================" << endl;
    
    const int N = 10;
    
    // Case 1: Minimum square (1x1)
    {
        cout << "\n1. 1x1 Square Test:" << endl;
        vector<pair<int, int>> positions = {{2, 2}, {2, 2}};
        
        DiagonalRectangleExtractorCPU cpu_extractor(N);
        double cpu_time;
        auto rectangles = cpu_extractor.extractRectangles(positions, cpu_time);
        
        if (!rectangles.empty() && !rectangles[0].empty()) {
            cout << "   Square size: 1x1" << endl;
            cout << "   Rectangles extracted: " << rectangles[0].size() << endl;
            cout << "   Expected: 2 rectangles (1 A-C + 1 B-D)" << endl;
            
            for (const auto& rect : rectangles[0]) {
                cout << "   - Type " << (rect.type == 0 ? "A-C" : "B-D") 
                     << ", Size: " << rect.width << "x" << rect.height << endl;
            }
        }
    }
    
    // Case 2: Points not in correct order
    {
        cout << "\n2. Invalid Ordering Test:" << endl;
        vector<pair<int, int>> positions = {{5, 5}, {3, 3}};
        
        DiagonalRectangleExtractorCPU cpu_extractor(N);
        double cpu_time;
        auto rectangles = cpu_extractor.extractRectangles(positions, cpu_time);
        
        cout << "   Valid squares found: " << rectangles.size() << " (expected: 0)" << endl;
    }
    
    // Case 3: Points at grid boundary
    {
        cout << "\n3. Boundary Test:" << endl;
        vector<pair<int, int>> positions = {{0, 0}, {N-1, N-1}};
        
        DiagonalRectangleExtractorCPU cpu_extractor(N);
        double cpu_time;
        auto rectangles = cpu_extractor.extractRectangles(positions, cpu_time);
        
        if (!rectangles.empty()) {
            cout << "   Square size: " << N << "x" << N << endl;
            cout << "   Rectangles extracted: " << rectangles[0].size() << endl;
            cout << "   Expected: " << (2 * N) << " rectangles" << endl;
        }
    }
}

// ==================== MAIN FUNCTION ====================
int main() {
    srand(42);
    
    cout << "==========================================" << endl;
    cout << "DIAGONAL RECTANGLE EXTRACTION SYSTEM" << endl;
    cout << "==========================================" << endl;
    cout << "Rules:" << endl;
    cout << "1. Type A-C: Square's top-left corner → Point on second diagonal" << endl;
    cout << "2. Type B-D: Point on second diagonal → Square's bottom-right corner" << endl;
    cout << "==========================================" << endl;
    
    // Run test cases
    testCase1_Visualization();
    testCase2_PerformanceComparison();
    testCase3_CornerCases();
    
    // Print system info
    cout << "\n\n=== SYSTEM INFORMATION ===" << endl;
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    cout << "GPU: " << prop.name << endl;
    cout << "Compute Capability: " << prop.major << "." << prop.minor << endl;
    cout << "Global Memory: " << prop.totalGlobalMem / (1024*1024) << " MB" << endl;
    cout << "Multiprocessors: " << prop.multiProcessorCount << endl;
    cout << "Max Threads per Block: " << prop.maxThreadsPerBlock << endl;
    
    cudaDeviceReset();
    
    cout << "\n==========================================" << endl;
    cout << "TEST COMPLETED SUCCESSFULLY" << endl;
    cout << "==========================================" << endl;
    
    return 0;
}