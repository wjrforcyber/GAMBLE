
#include "sectionProcessGPU.hpp"

int main() {
    const int MATRIX_ROWS = 10;
    const int MATRIX_COLS = 10;

    // Create source matrix
    std::vector<int> h_matrix(MATRIX_ROWS * MATRIX_COLS);
    for (int i = 0; i < MATRIX_ROWS * MATRIX_COLS; i++) {
        h_matrix[i] = static_cast<int>(rand()) % 10;
    }

    //print out the original matrix
    std::cout << "Original Matrix:" << std::endl;
    for (int i = 0; i < MATRIX_ROWS; i++) {
        for (int j = 0; j < MATRIX_COLS; j++) {
            std::cout << h_matrix[i * MATRIX_COLS + j] << " ";
        }
        std::cout << std::endl;
    }

    // Define sections to process
    std::vector<MatrixSection> sections = {
        {1, 1, 3, 3},  // Section 1
        {2, 2, 7, 7},  // Section 2  
        {1, 3, 8, 6},  // Section 3
        {2, 4, 5, 9}   // Section 4
    };
    
    // Allocate device memory for source matrix
    int* d_matrix;
    cudaMalloc(&d_matrix, MATRIX_ROWS * MATRIX_COLS * sizeof(int));
    cudaMemcpy(d_matrix, h_matrix.data(), MATRIX_ROWS * MATRIX_COLS * sizeof(int),
               cudaMemcpyHostToDevice);
    
    // Process sections
    MatrixSectionProcessor processor(sections);
    processor.processSections(d_matrix, MATRIX_ROWS, MATRIX_COLS);
    
    // Copy result back
    cudaMemcpy(h_matrix.data(), d_matrix, MATRIX_ROWS * MATRIX_COLS * sizeof(int),
               cudaMemcpyDeviceToHost);
    //print out the processed matrix
    std::cout << "Processed Matrix:" << std::endl;
    for (int i = 0; i < MATRIX_ROWS; i++) {
        for (int j = 0; j < MATRIX_COLS; j++) {
            std::cout << h_matrix[i * MATRIX_COLS + j] << " ";
        }
        std::cout << std::endl;
    }
    cudaFree(d_matrix);
    return 0;
}