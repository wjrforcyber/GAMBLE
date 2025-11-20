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
    pair<int,int> source = {0,0};
    pair<int,int> sink = {0,3};
    squareN = abs(source.first - sink.first) + 1 > abs(source.second - sink.second) + 1 ? abs(source.first - sink.first) + 1 : abs(source.second - sink.second) + 1;
    auto indices = getSecDiagMatrixIndices(make_pair(0, 0), squareN);

    std::vector<MatrixSection> sections;
    // TODO@ Jingren Wang: (Bottom, Right) should be the flipped (Bottom, Right)
    createSectionsOnSecDiag(indices, sections,make_pair(0, 0), make_pair(squareN - 1, squareN - 1));
    
    assert(sections.size() == 8); // For a 10x10 matrix, there should be 10 sections
    vector<vector<int>> originalMatrix = {{30,32,421,40}, 
                                          {1,5,7,23},
                                          {23,65,76,23},
                                          {21,54,23,8}};
    
    vector<pair<int, int>> cpures;
    vector<pair<int, int>> pins = {make_pair(0, 0), make_pair(0, 3)};
    MazeRouter mazeRouter;
    auto cputime = mazeRouter.route(originalMatrix, 4, pins, cpures);
    //show the path
    cout << "Original maze route Path: ";
    for (auto [r, c] : cpures) {
        cout << "(" << r << "," << c << ") ";
    }
    std::cout << std::endl;
    std::cout << "Cost is " << evaluate(cpures, originalMatrix, 4);
    std::cout << std::endl;
    
    //// Process sections on CPU
    TimeProfile t;
    MatrixSectionProcessorCPU processor(originalMatrix, sections);
    pair<int, int> rSource = make_pair(0, 0);
    pair<int, int> rSink = make_pair(0, 3);
    processor.getMinCostAndPathOnSections(originalMatrix, rSource, rSink, &t);
    
    //// Evaluate unconnected pins from 4 direction
    vector<pair<int, int>> uncPins = {};
    // Calculate each pair of the path and find the optimum one
    PathInfo info = processor.selectFromMinCostAndPath(sections, originalMatrix, rSource, rSink, uncPins);
    // Print the final cost and the path selected
    std::cout << "The final cost is " << info.cost << ". " << std::endl;
    std::cout << "The final path is " << std::endl;
     //show the path
    cout << "Path: ";
    for (auto [r, c] : info.path) {
        cout << "(" << r << "," << c << ") ";
    }
    assert(evaluate(cpures, originalMatrix, 4) < info.cost);
    std::cout << std::endl;
    std::cout << std::endl;
    
    return 0;
}