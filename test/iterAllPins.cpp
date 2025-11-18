#include "util.hpp"
#include <cassert>
#include <iostream>
#include <utility>
#include "sectionProcessCPU.hpp"
#include "mazeRouter.hpp"

/*!
  \brief Let's give a matrix that has size of 10 * 10. And give 4 sparse pins.
  Check all pins will be connected to the single net.
*/
const int N = 10;
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


/*!
  \brief This test case use a simple method to make sure at least every pin is connected to a single net.
*/
int main() {
    vector<pair<int, int>> pins = {make_pair(0, 1), make_pair(1, 3), make_pair(3, 2), make_pair(8,9)};
    // Create source matrix
    vector<vector<int>> h_matrix(N, vector<int> (N));
    for (int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++)
            h_matrix[i][j] = static_cast<int>(rand()) % 10;
    }
    int squareN;
    auto rh_matrix =  h_matrix;

    vector<pair<int, int>> cpures;
    MazeRouter mazeRouter;
    auto cputime = mazeRouter.route(h_matrix, N, pins, cpures);
    //show the path
    cout << "Original maze route Path: ";
    for (auto [r, c] : cpures) {
        cout << "(" << r << "," << c << ") ";
    }
    cout << endl;
    cout << "Cost is " << evaluate(cpures, h_matrix, N);
    cout << endl;

    DazeRouter dazeRouter;
    vector<pair<int, int>> dres;
    auto dTime = dazeRouter.route(rh_matrix, N, pins, dres);
    assert(checkAllPinsOnPath(dres, pins));
    // Print the final cost and the path selected
    cout << "The final cost is " << evaluate(dres, rh_matrix, N) << ". " << endl;
    cout << "The final path is " << endl;
     //show the path
    cout << "Path: ";
    for (auto [r, c] : dres) {
        cout << "(" << r << "," << c << ") ";
    }
    cout << endl;
    cout << endl;
    return 0;
}