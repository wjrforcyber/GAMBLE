#include "util.hpp"
#include <cassert>
#include <iostream>
#include <utility>
#include "sectionProcessCPU.hpp"

int main() {
    // Seleted N should be decided by the selected large square
    auto LU = make_pair(0, 0);
    auto RD = make_pair(4,4);
        //cout << "Selected suqare N is " << squareN << endl;
    
    int squareN;
    // The indices must has real source and sink as para
    squareN = abs(LU.first - RD.first) + 1 > abs(LU.second - RD.second) + 1 ? abs(LU.first - RD.first) + 1 : abs(LU.second - RD.second) + 1;
    auto indices = getSecDiagMatrixIndices(LU, RD, squareN);
    //show the indices
    for (auto [l, r] : indices) {
        cout << "(" << l << "," << r << ") ";
    }
    // Create source matrix
    std::vector<int> h_matrix(squareN * squareN);
    for (int i = 0; i < squareN * squareN; i++) {
        h_matrix[i] = static_cast<int>(rand()) % 10;
    }

    //print out the original matrix
    std::cout << "Original Matrix:" << std::endl;
    for (int i = 0; i < squareN; i++) {
        for (int j = 0; j < squareN; j++) {
            std::cout << h_matrix[i * squareN + j] << " ";
        }
        std::cout << std::endl;
    }
    std::vector<MatrixSection> sections;
    createSectionsOnSecDiag(indices, sections, make_pair(0, 0), make_pair(4, 4));
    
    assert(sections.size() == 10); // For a 10x10 matrix, there should be 10 sections
    
    // Convert flat matrix to 2D vector
    std::vector<std::vector<int>> originalMatrix(squareN, std::vector<int>(squareN));
    for (int i = 0; i < squareN; i++) {
        for (int j = 0; j < squareN; j++) {
            originalMatrix[i][j] = h_matrix[i * squareN + j];
        }
    }
    // Process sections on CPU
    MatrixSectionProcessorCPU processor(originalMatrix, sections);
    processor.getMinCostAndPathOnSections(originalMatrix, make_pair(0, 0), make_pair(4, 4));
    return 0;
}