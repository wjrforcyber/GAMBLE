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
    auto indices = getSecDiagMatrixIndices(make_pair(0, 1), make_pair(1,3), squareN);

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

    pair<int, int> source = {0,1};
    // Convert flat matrix to 2D vector
    vector<vector<int>> originalMatrix(squareN, vector<int>(squareN));
    for (int i = 0; i < squareN; i++) {
        for (int j = 0; j <squareN; j++) {
            //should add source node shift here
            originalMatrix[i][j] = h_matrix[(i + source.first) * N + (j + source.second)];
        }
    }
    
    cout << "Converted cost matrix "<< endl;
    for (int i = 0; i < squareN; i++) {
        for (int j = 0; j < squareN; j++) {
            cout << originalMatrix[i][j] << "(" << vOriginalMatrix[i + source.first][j + source.second] << ")";
            assert(originalMatrix[i][j] == vOriginalMatrix[i + source.first][j + source.second]);
        }
        cout << endl;
    }

    return 0;
}