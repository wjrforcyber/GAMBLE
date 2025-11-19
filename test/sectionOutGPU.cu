#include "sectionProcessGPU.hpp"
int main() {
    const int MATRIX_ROWS = 10;
    const int MATRIX_COLS = 10;

    // Create source matrix
    std::vector<int> h_matrix(MATRIX_ROWS * MATRIX_COLS);
    for (int i = 0; i < MATRIX_ROWS * MATRIX_COLS; i++) {
        h_matrix[i] = static_cast<int>(rand()) % 10;
    }

    // Print original matrix
    std::cout << "Original Matrix:" << std::endl;
    for (int i = 0; i < MATRIX_ROWS; i++) {
        for (int j = 0; j < MATRIX_COLS; j++) {
            std::cout << h_matrix[i * MATRIX_COLS + j] << " ";
        }
        std::cout << std::endl;
    }

    // Define sections to process (note: these are 0-indexed)
    std::vector<MatrixSection> sections = {
        {1, 1, 3, 3},  // Section 1: 2x2 section (rows 1-2, cols 1-2)
        {2, 2, 7, 7},  // Section 2: 5x5 section (rows 2-6, cols 2-6)
        {1, 3, 8, 6},  // Section 3: 7x3 section (rows 1-7, cols 3-5)  
        {2, 4, 5, 9}   // Section 4: 3x5 section (rows 2-4, cols 4-8)
    };
    
    // Allocate device memory for source matrix
    int* d_matrix;
    cudaMalloc(&d_matrix, MATRIX_ROWS * MATRIX_COLS * sizeof(int));
    cudaMemcpy(d_matrix, h_matrix.data(), MATRIX_ROWS * MATRIX_COLS * sizeof(int),
               cudaMemcpyHostToDevice);
    
    // Process sections
    MatrixSectionProcessor processor(sections);
    processor.processSections(d_matrix, MATRIX_ROWS, MATRIX_COLS);
    
    std::cout << "\n=== Sections After Processing ===" << std::endl;
    processor.printSections();
    
    // Copy result back to main matrix
    cudaMemcpy(h_matrix.data(), d_matrix, MATRIX_ROWS * MATRIX_COLS * sizeof(int),
               cudaMemcpyDeviceToHost);
    
    // Print final matrix
    std::cout << "\n=== Final Matrix ===" << std::endl;
    for (int i = 0; i < MATRIX_ROWS; i++) {
        for (int j = 0; j < MATRIX_COLS; j++) {
            std::cout << h_matrix[i * MATRIX_COLS + j] << " ";
        }
        std::cout << std::endl;
    }
    
    cudaFree(d_matrix);
    return 0;
}