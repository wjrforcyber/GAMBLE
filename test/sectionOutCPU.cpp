
#include "sectionProcessCPU.hpp"
#include <cassert>
#include <utility>

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
    // Convert flat matrix to 2D vector
    std::vector<std::vector<int>> originalMatrix(MATRIX_ROWS, std::vector<int>(MATRIX_COLS));
    for (int i = 0; i < MATRIX_ROWS; i++) {
        for (int j = 0; j < MATRIX_COLS; j++) {
            originalMatrix[i][j] = h_matrix[i * MATRIX_COLS + j];
        }
    }
    // Process sections on CPU
    MatrixSectionProcessorCPU processor(originalMatrix, sections);
    processor.getMinCostAndPathOnSections(originalMatrix, make_pair(0,  0), make_pair(3, 3));
    //verify the results with assertion
    auto processedSections = processor.getSections();
    for (size_t s = 0; s < sections.size(); s++) {
        MatrixSection& sec = processedSections[s];
        for (int r = 0; r < sec.rows; r++) {
            for (int c = 0; c < sec.cols; c++) {
                int expectedValue = originalMatrix[sec.top + r][sec.left + c];
                assert(sec.data[r][c] == expectedValue);
            }
        }
    }
    return 0;
}