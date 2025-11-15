#include "util.hpp"
#include <cassert>
#include <iostream>
#include <utility>
#include "sectionProcessCPU.hpp"
#include "mazeRouter.hpp"

const int N = 5;
//, NumBlks, NumPins;
const int dx[] = {-1, 1, 0, 0};
const int dy[] = {0, 0, -1, 1};

void dfs(int x, int y, vector<vector<int>> &cost) {
    if(cost[x][y] != -1) return;
    cost[x][y] = 0;
    for(int d = 0; d < 4; d++) {
        int nx = x + dx[d], ny = y+ dy[d];
        if(0 <= nx && nx < N && 0 <= ny && ny < N)
            dfs(nx, ny, cost);
    }
}
int evaluate(const vector<pair<int, int>> &res, vector<vector<int>> cost, const int N) {   
    const int turnCost = 50; 
    int tot = 0;
    for(auto e : res) {
        assert(cost[e.first][e.second] != -1);
        tot += cost[e.first][e.second];
        cost[e.first][e.second] = -1;
    }
    
    int turnCount = 0;
    for(int i = 0; i < N; i++)
        for(int j = 0; j < N; j++) if(cost[i][j] == -1) {
            int test[2] = {0, 0};
            for(int d = 0; d < 4; d++) {
                int x = i + dx[d], y = j + dy[d];
                test[d / 2] |= (0 <= x && x < N && 0 <= y && y < N && cost[x][y] == -1);
            }
            turnCount += test[0] && test[1];
        }
    dfs(res[0].first, res[0].second, cost);
    tot += turnCount * turnCost;
    
    for(auto e : res) 
        if(cost[e.first][e.second] == -1) tot = -1;
    return tot;
}

int main() {
    int squareN;
    // The indices must has real source and sink as para
    pair<int,int> source = {0,0};
    pair<int,int> sink = {4,4};
    squareN = abs(source.first - sink.first) + 1 > abs(source.second - sink.second) + 1 ? abs(source.first - sink.first) + 1 : abs(source.second - sink.second) + 1;
    auto indices = getSecDiagMatrixIndices(make_pair(0, 0), squareN);

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
    createSectionsOnSecDiag(indices, sections,make_pair(0, 0), make_pair(4, 4));
    
    assert(sections.size() == 10); // For a 10x10 matrix, there should be 10 sections
    
    // Convert flat matrix to 2D vector
    std::vector<std::vector<int>> originalMatrix(squareN, std::vector<int>(squareN));
    for (int i = 0; i < squareN; i++) {
        for (int j = 0; j <squareN; j++) {
            originalMatrix[i][j] = h_matrix[i * squareN + j];
        }
    }
    vector<pair<int, int>> cpures;
    vector<pair<int, int>> pins = {make_pair(0, 0), make_pair(4, 4)};
    MazeRouter mazeRouter;
    auto cputime = mazeRouter.route(originalMatrix, 5, pins, cpures);
    //show the path
    cout << "Original maze route Path: ";
    for (auto [r, c] : cpures) {
        cout << "(" << r << "," << c << ") ";
    }
    std::cout << std::endl;
    std::cout << "Cost is " << evaluate(cpures, originalMatrix, 5);
    std::cout << std::endl;
    
    // Process sections on CPU
    MatrixSectionProcessorCPU processor(originalMatrix, sections);
    pair<int, int> rSource = make_pair(0, 0);
    pair<int, int> rSink = make_pair(4, 4);
    processor.getMinCostAndPathOnSections(originalMatrix, rSource, rSink);
    
    // Calculate each pair of the path and find the optimum one
    vector<pair<int, int>> uncPins = {};
    PathInfo info = processor.selectFromMinCostAndPath(sections, originalMatrix, rSource, rSink, uncPins);
    // Print the final cost and the path selected
    std::cout << "The final cost is " << info.cost << ". " << std::endl;
    std::cout << "The final path is " << std::endl;
     //show the path
     cout << "Path: ";
     for (auto [r, c] : info.path) {
         cout << "(" << r << "," << c << ") ";
     }
     std::cout << std::endl;
     std::cout << std::endl;
    
    return 0;
}