
#include <iostream>
#include <utility>
#include <vector>
#include "mazeRouter.hpp"

#include <thrust/device_vector.h>
#include <thrust/reduce.h>

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
/*
This is a example of testing bi-directional partition based paralled maze router.
*/


using namespace std;
/*!
 \brief This should be align with the assignment format with N * N
*/

int initCostTestCase(bool iLarge)
{
    // TODO @Jingren: implement different test cases
    return 0;
}


/*!
  \brief Sum up the matrix using Thrust library on GPU
*/
int sumMatrix(const std::vector<std::vector<int>>& matrix) {
    // Flatten the 2D vector to 1D
    std::vector<int> flattened;
    for (const auto& row : matrix) {
        flattened.insert(flattened.end(), row.begin(), row.end());
    }
    
    // Copy to device and sum
    thrust::device_vector<int> d_vec = flattened;
    return thrust::reduce(d_vec.begin(), d_vec.end());
}

/*!
  \brief Normalize the matrix. Substract each element with the minimum value in the matrix.
*/
int normalizeCostMatrix(vector<vector<int>> &cost, int N)
{
    int minValue = cost[0][0];
    for(int i = 0; i < N; i++)
    {
        for(int j = 0; j < N; j++)
        {
            if(cost[i][j] < minValue)
                minValue = cost[i][j];
        }
    }

    for(int i = 0; i < N; i++)
    {
        for(int j = 0; j < N; j++)
        {
            cost[i][j] -= minValue;
        }
    }
    return minValue;
}


/*!
 \brief Compute the sum of each row and sum of each column
*/
void computeRowColSum(const vector<vector<int>> &cost, vector<int> &rowSum, vector<int> &colSum, int N)
{
    for(int i = 0; i < N; i++)
    {
        int rSum = 0;
        for(int j = 0; j < N; j++)
        {
            rSum += cost[i][j];
        }
        rowSum[i] = rSum;
    }

    for(int j = 0; j < N; j++)
    {
        int cSum = 0;
        for(int i = 0; i < N; i++)
        {
            cSum += cost[i][j];
        }
        colSum[j] = cSum;
    }
}

/*!
  \brief Check if the pins are valid, should between 0 and N-1
*/
bool checkPinsValidity(const pair<int, int> &pin, int N)
{
    if(pin.first < 0 || pin.first >= N || pin.second < 0 || pin.second >= N)
        return false;
    return true;
}

/*!
  \brief Collect corner case up/down with single via
*/
pair<vector<pair<int, int>>,int >collectUp(const vector<vector<int>> &cost, int costSum, vector<pair<int, int>> &res)
{
    //collect from (0,0) to (0,N-1)
    for(int j = 0; j < N; j++)
    {
        res.emplace_back(make_pair(0, j));
    }
    //collect from (1, N-1) to (N-1, N-1)
    for(int i = 1; i < N; i++)
    {
        res.emplace_back(make_pair(i, N-1));
    }
    return make_pair(res, costSum);
}
pair<vector<pair<int, int>>,int >collectDown(const vector<vector<int>> &cost, int costSum, vector<pair<int, int>> &res)
{
    
    //collect from (0,0) to (N - 1, 0)
    for(int i = 0; i < N; i++)
    {
        res.emplace_back(make_pair(i, 0));
    }
    //collect from (N-1, 1) to (N-1, N-1)
    for(int j = 1; j < N; j++)
    {
        res.emplace_back(make_pair(N-1, j));
    }
    return make_pair(res, costSum);
}

/*!
  \brief Label the special cases based on the cost sum and via cost
*/
pair<vector<pair<int, int>>,int > caseLabelHandling(const vector<vector<int>> &cost, int costSum, int viaCost)
{
    vector<int> rowSum(N);
    vector<int> colSum(N);
    vector<pair<int, int>> res;
    //pair<vector<pair<int, int>>,int > res;
    computeRowColSum(cost, rowSum, colSum, N);
    if(costSum < 2 * viaCost)
    {
        int up = rowSum[0] + colSum[N-1] - cost[0][N-1];
        int down = rowSum[N-1] + colSum[0] - cost[N-1][0];
        return up < down ? collectUp(cost, up, res) : collectDown(cost, down, res);
    }
    //else if(costSum >= viaCost && costSum < 2 * viaCost)
    //    ; //case 2
    //else
        //return 0; //normal case
}
/*!
  \brief Some special case handling, e.g., 
  1. the sum of the values in a rectangle is lower than a via;
  2. the sum of the values in a rectangle is in [1,2) times via cost.
  \param pin1 The first pin, left top corner
  \param pin2 The second pin, right bottom corner
  \param cost The cost matrix
*/
pair<vector<pair<int, int>>,int > caseSpecial(const pair<int, int> &pin1, const pair<int, int> &pin2, const vector<vector<int>> &cost, const int viaCost)
{
    int mSum = 0;
    // TODO @Jingren: Give threshold and decide when to use CPU and when to use GPU
    for(int i = pin1.first; i <= pin2.first; i++)
    {
        for(int j = pin1.second; j <= pin2.second; j++)
        {
            //sum up the cost in the rectangle
            mSum += cost[i][j];
        }
    }
    cout << "costSum: " << mSum << ", viaCost: " << viaCost << endl;
    return caseLabelHandling(cost, mSum, viaCost);
}


/*!
 \brief A simple case between 2 pins
*/
int SimpleCaseBetween2Pins(pair<int, int> pin1, pair<int, int> pin2, vector<vector<int>> &cost, int N, vector<vector<pair<int, int>>> &paths)
{
    return 0;
}


/*!
  \brief Show the path of the final result.
*/
void showPathResult(const vector<pair<int, int>> &res)
{
    cout << "The path is: " << endl;
    for(auto e : res)
        cout << "(" << e.first << ", " << e.second << ") ";
    cout << endl;
}


int main()
{
    //init a cost matrix that has random values between 0-9 in costMH
    int costMH[5][5] = {
        {1, 1, 5, 2, 4},
        {5, 1, 2, 3, 9},
        {6, 3, 5, 9, 9},
        {1, 2, 3, 5, 8},
        {9, 6, 1, 3, 2}
    };

    vector<vector<int>> cost(5, vector<int>(5, 0));
    //initialize cost matrix
    for(int i = 0; i < 5; i++)
    {
        for(int j = 0; j < 5; j++)
        {
            cost[i][j] = costMH[i][j];
        }
    }
    
    //preprocess the matrix, normalize it
    int miniVal = normalizeCostMatrix(cost, N);
    vector<pair<int, int>> cpures;
    vector<pair<int, int>> pins = {make_pair(0, 0), make_pair(4, 4)};
    MazeRouter mazeRouter;
    auto cputime = mazeRouter.route(cost, 5, pins, cpures);
    showPathResult(cpures);
    cout << "CPU version:\n" <<  "    time: " << cputime.first * 1.0 / CLOCKS_PER_SEC << "s\n    cost: " << evaluate(cpures, cost, 5) + cpures.size() *  miniVal << endl;
    
    auto clocks = clock();
    auto caseSp = caseSpecial(make_pair(0, 0), make_pair(4, 4), cost, turnCost);
    clocks = clock() - clocks;
    showPathResult(caseSp.first);
    cout << "Special case handling:\n    time: " << clocks * 1.0 / CLOCKS_PER_SEC << "s\n    cost: " << evaluate(caseSp.first, cost, 5) + caseSp.first.size() * miniVal << endl;
   
    //special case test
    return 0;
}