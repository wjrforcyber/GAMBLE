#include "util.hpp"
#include <cassert>
#include <iostream>
#include <utility>
#include "sectionProcessCPU.hpp"

const int N = 5;

/*!
  \brief This test case just check the via Map should have the same length as the path, since one location is unique on key.
*/
int main() {
    int squareN;
    pair<int,int> source = {0,0};
    pair<int,int> sink = {1,3};
    squareN = abs(source.first - sink.first) + 1 > abs(source.second - sink.second) + 1 ? abs(source.first - sink.first) + 1 : abs(source.second - sink.second) + 1;
    auto indices = getSecDiagMatrixIndices(make_pair(0, 0), squareN);

    // Create source matrix
    std::vector<int> h_matrix(N * N);
    for (int i = 0; i < N * N; i++) {
        h_matrix[i] = static_cast<int>(rand()) % 10;
    }

    std::vector<MatrixSection> sections;
    // TODO@ Jingren Wang: (Bottom, Right) should be the flipped (Bottom, Right)
    createSectionsOnSecDiag(indices, sections,make_pair(0, 0), make_pair(squareN - 1, squareN - 1));
    
    assert(sections.size() == 8); // For a 10x10 matrix, there should be 10 sections
    
    // Convert flat matrix to 2D vector
    std::vector<std::vector<int>> originalMatrix(squareN, std::vector<int>(squareN));
    for (int i = 0; i < squareN; i++) {
        for (int j = 0; j <squareN; j++) {
            originalMatrix[i][j] = h_matrix[i * N + j];
        }
    }
    // Process sections on CPU
    MatrixSectionProcessorCPU processor(originalMatrix, sections);
    pair<int, int> rSource = make_pair(0, 0);
    pair<int, int> rSink = make_pair(1, 3);
    processor.getMinCostAndPathOnSections(originalMatrix, rSource, rSink );
    
    // Evaluate unconnected pins from 4 direction
    vector<pair<int, int>> uncPins = {{3,2}};
    // Calculate each pair of the path and find the optimum one
    PathInfo info = processor.selectFromMinCostAndPath(sections, originalMatrix, rSource, rSink, uncPins);
    assert(info.path.size() == info.isLocVia.size());
    cout << "Show the key of the isLocVia: "<<endl;
    for(auto &item: info.isLocVia)
    {
        cout << "(" << item.first.first << "," << item.first.second << ")" << " " << item.second << endl;
    }
    return 0;
}