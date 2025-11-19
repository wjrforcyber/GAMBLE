#include "mazeRouter.hpp"
#include "sectionProcessCPU.hpp"
#include <iostream>

int N, NumBlks, NumPins;
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
    {
        if(cost[e.first][e.second] == -1)
        {
            cout << "Encounter dangling (" << e.first << "," << e.second << ")" <<endl;
            tot = -1;
        }
    }
        
    return tot;
}

int main() {
    
    //cout << "Number of grid cells (N x N): " << endl;
    cin >> N;
    vector<vector<int>> cost(N, vector<int> (N));
    //cout << "Cost matrix (" << N << " x " << N << "): " << endl;
    for(int i = 0; i < N; i++)
        for(int j = 0; j < N; j++)
            cin >> cost[i][j];
    //cout << "Give the number of blocks that cannot be routed through: " << endl;
    cin >> NumBlks;
    for(int i = 0; i < NumBlks; i++) {
        int x1, y1, x2, y2;
        //cout << "Block " << i + 1 << " (x1 y1 x2 y2): " << endl;
        cin >> x1 >> y1 >> x2 >> y2;
        for(int a = x1; a <= x2; a++)
            for(int b = y1; b <= y2; b++)
                cost[a][b] = INF;
    }
    //cout << "Give the number of pins to be connected: " << endl;
    cin >> NumPins;
    vector<pair<int, int>> pins(NumPins), gpures, cpures;
    for(int i = 0; i < NumPins; i++)
    {
        //cout << "Pins (x y): " << endl;
        cin >> pins[i].first >> pins[i].second;
    }
    vector<vector<int>> costD(N, vector<int> (N));
    costD = cost;
    MazeRouter mazeRouter;
    auto cputime = mazeRouter.route(cost, N, pins, cpures);
    
    DazeRouter dazeRouter;
    vector<pair<int, int>> cpuresD;
    auto cputimeD = dazeRouter.route(costD, N, pins, cpuresD);
    assert(checkAllPinsOnPath(cpuresD, pins));
    cout << "Original CPU version:\n" <<  "    time: " << cputime.first * 1.0 / CLOCKS_PER_SEC << "s\n    cost: " << evaluate(cpures, cost, N) << endl;
    cout << "Current CPU version:\n" <<  "    time: " << cputimeD.first * 1.0 / CLOCKS_PER_SEC << "s\n    cost: " << evaluate(cpuresD, costD, N) << endl;

    return 0;
}