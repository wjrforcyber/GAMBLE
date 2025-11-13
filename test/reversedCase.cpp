#include "util.hpp"
#include <cassert>
#include <iostream>
#include <utility>
#include "sectionProcessCPU.hpp"

const int N = 5;

int main() {
    int squareN;
    // The indices must has real source and sink as para
    pair<int,int> source = {0,4};
    pair<int,int> sink = {4,0};
    squareN = abs(source.first - sink.first) + 1 > abs(source.second - sink.second) + 1 ? abs(source.first - sink.first) + 1 : abs(source.second - sink.second) + 1;
    assert(squareN == 5);

    //This should provide exact value of the LU and RL position
    auto indices = getSecDiagMatrixIndices(make_pair(0, 0), make_pair(squareN - 1, squareN - 1), squareN);

    //Fliped real source
    pair<int, int> sourceR = {source.first, (squareN - 1) - source.second};
    pair<int, int> sinkR = {sink.first, (squareN - 1) - sink.second};
    assert(sourceR.first == 0 && sourceR.second == 0);
    cout << "Sink: " << sinkR.first << " " << sinkR.second << endl;
    assert(sinkR.first == 4 && sinkR.second == 4);

    // Create source matrix
    vector<int> h_matrix(N * N);
    for (int i = 0; i < N * N; i++) {
        h_matrix[i] = static_cast<int>(rand()) % 10;
    }

    //print out the original matrix
    cout << "Very Original Matrix:" << endl;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            cout << h_matrix[i * N + j] << " ";
        }
        cout << endl;
    }
    vector<vector<int>> vOriginalMatrix(N, vector<int>(N));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            //should add source node shift here
            vOriginalMatrix[i][j] = h_matrix[i * N + j];
        }
    }
    vector<MatrixSection> sections;
    // Create Section should have (0,0) and (squareN - 1, squareN -1).
    createSectionsOnSecDiag(indices, sections,make_pair(0, 0), make_pair(squareN - 1, squareN - 1));
    assert(sections.size() == 10);

    // Convert flat matrix to 2D vector
    vector<vector<int>> originalMatrix(squareN, vector<int>(squareN));
    for (int i = 0; i < squareN; i++) {
        for (int j = 0; j <squareN; j++) {
            //should add source node shift here
            originalMatrix[i][j] = h_matrix[(i + sourceR.first) * N + (j + sourceR.second)];
        }
    }
    
    cout << "Converted cost matrix "<< endl;
    for (int i = 0; i < squareN; i++) {
        for (int j = 0; j < squareN; j++) {
            cout << originalMatrix[i][j] << "(" << vOriginalMatrix[i + sourceR.first][j + sourceR.second] << ")";
            assert(originalMatrix[i][j] == vOriginalMatrix[i + sourceR.first][j + sourceR.second]);
        }
        cout << endl;
    }
    
    MatrixSectionProcessorCPU processor(originalMatrix, sections);
    processor.getMinCostAndPathOnSections(originalMatrix, sourceR, sinkR );
    vector<pair<int, int>> uncPins = {};
    PathInfo info = selectFromMinCostAndPath(sections, originalMatrix, sourceR, sinkR, uncPins);
    assert(info.path[0] == sourceR && info.path[info.path.size() - 1] == sinkR);
    //Reverse the path, use squareN - i do the reverse
    info.reverseInplace(squareN);
    assert(info.path[0] == source && info.path[info.path.size() - 1] == sink);
    cout << " The final cost is " << info.cost << ", and the final path is " << endl;
    for(auto pos: info.path)
    {
        cout << "(" << pos.first << "," << pos.second << ")";
    }
    cout << endl;
    return 0;
}