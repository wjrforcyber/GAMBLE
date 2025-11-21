#ifndef SECTION_OPS_CPU_HPP
#define SECTION_OPS_CPU_HPP
#include <algorithm>
#include <cassert>
#include <climits>
#include <iostream>
#include <type_traits>
#include <utility>
#include <vector>
#include "mazeRouter.hpp"
#include "pathInfo.hpp"
#include "util.hpp"

using namespace std;

class MatrixSectionProcessorCPU {
public:
    MatrixSectionProcessorCPU(vector<vector<int>>& originalMatrix, vector<MatrixSection>& sections) 
        : sections(sections), numSections(sections.size()), originalMatrix(originalMatrix) {
        // Initialize section dimensions
        for (int i = 0; i < numSections; i++) {
            sections[i].rows = sections[i].bottom - sections[i].top + 1;
            sections[i].cols = sections[i].right - sections[i].left + 1;
            // put the data into section.data
            sections[i].data.resize(sections[i].rows, vector<int>(sections[i].cols));
            for (int r = 0; r < sections[i].rows; r++) {
                for (int c = 0; c < sections[i].cols; c++) {
                    sections[i].data[r][c] = originalMatrix[sections[i].top + r][sections[i].left + c];
                }
            }
        }
    }
    
    //show all sections
    void getMinCostAndPathOnSections(vector<vector<int>>& originalMatrix, pair<int, int> realSource, pair<int, int> realSink) {
        for (int i = 0; i < numSections; i++) {
            cout << "Section " << i << " (" << sections[i].rows << "x" << sections[i].cols << "):" << endl;
            for (int r = 0; r < sections[i].rows; r++) {
                for (int c = 0; c < sections[i].cols; c++) {
                    cout << sections[i].data[r][c] << " ";
                }
                cout << endl;
            }
            //also show the shortest path sum with turn cost
            auto [minCost, path, isLocVia] = sections[i].shortestPathSumWithTurnCost();
            cout << "Original Path Sum with Turn Cost: " << minCost << endl;
            //show the path
            cout << "Path: ";
            for (auto [r, c] : path) {
                cout << "(" << r << "," << c << ") ";
            }
            cout << endl;
            auto [minCostUpdate, pathUpdate, isLocViaUpdate] = realValCheck(originalMatrix, realSink, path, isLocVia);

            cout << "Shortest Path Sum with Turn Cost: " << minCostUpdate << endl;
            //show the path
            cout << "Path: ";
            for (auto [r, c] : pathUpdate) {
                cout << "(" << r << "," << c << ") ";
            }
            // pass the minCost and path to section[i]
            sections[i].minCost = minCostUpdate;
            sections[i].bPath = pathUpdate;
            sections[i].isLocVia = isLocViaUpdate;
            cout << endl;
            cout << endl;
        }
    }
    
    //get sections
    vector<MatrixSection>& getSections() {
        return sections;
    }
    
    PathInfo creatAInfoOnFly(vector<std::vector<int>>& originalMatrix, const pair<int, int>& rSource, const pair<int, int>& rSink)
    {
        PathInfo UpCreate;
        UpCreate.path.clear();
        UpCreate.cost = 0;
        UpCreate.isLocVia.clear();
        
        PathInfo DownCreate;
        DownCreate.path.clear();
        DownCreate.cost = 0;
        DownCreate.isLocVia.clear();
        
        bool fineOnUp = true;
        bool fineOnDown = true;
        //Up
        for(auto j = 0 ; j < rSink.second + 1; j++)
        {
            if(originalMatrix[0][j] == INF)
            {
                cout << "Failed on upper path." << endl;
                UpCreate.path.clear();
                UpCreate.cost = INF;
                UpCreate.isLocVia.clear();
                fineOnUp = false;
                break;
            }
            UpCreate.path.push_back(make_pair(0, j));
            UpCreate.cost += originalMatrix[0][j];
            if(j == rSink.second)
            {
                UpCreate.isLocVia[make_pair(0, j)] = true;
            }
            else {
                UpCreate.isLocVia[make_pair(0, j)] = false;
            }
        }
        if(fineOnUp)
        {
            for(auto i = 1 ; i < rSink.first + 1; i++)
            {
                if(originalMatrix[i][rSink.second] == INF)
                {
                    cout << "Failed on upper path." << endl;
                    UpCreate.path.clear();
                    UpCreate.cost = INF;
                    UpCreate.isLocVia.clear();
                    break;
                }
                UpCreate.path.push_back(make_pair(i, rSink.second));
                UpCreate.cost += originalMatrix[i][rSink.second];
                UpCreate.isLocVia[make_pair(i, rSink.second)] = false;
            }
        }
        
        //Down
        for(auto i = 0 ; i < rSink.first + 1; i++)
        {
            if(originalMatrix[i][0] == INF)
            {
                cout << "Failed on lower path." << endl;
                DownCreate.path.clear();
                DownCreate.cost = INF;
                DownCreate.isLocVia.clear();
                fineOnDown = false;
                break;
            }
            DownCreate.path.push_back(make_pair(i, 0));
            DownCreate.cost += originalMatrix[i][0];
            if(i == rSink.first)
            {
                DownCreate.isLocVia[make_pair(i, 0)] = true;
            }
            else {
                DownCreate.isLocVia[make_pair(i, 0)] = false;
            }
        }
        if(fineOnDown)
        {
            for(auto j = 1 ; j < rSink.second + 1; j++)
            {
                if(originalMatrix[rSink.first][j] == INF)
                {
                    cout << "Failed on lower path." << endl;
                    DownCreate.path.clear();
                    DownCreate.cost = INF;
                    DownCreate.isLocVia.clear();
                    break;
                }
                DownCreate.path.push_back(make_pair(rSink.first, j));
                DownCreate.cost += originalMatrix[rSink.first][j];
                DownCreate.isLocVia[make_pair(rSink.first, j)] = false;
            }
        }
        cout << "Downpath cost " << DownCreate.cost << " Uppath cost " << UpCreate.cost << endl;
        if(DownCreate.cost == INF && UpCreate.cost == INF)
        {
            cout << "Nooooo..." << endl;
        }
        if(DownCreate.cost < UpCreate.cost)
        {
            cout << "return lower path" << endl;
            return DownCreate;
        }
        return UpCreate;
    }
    
    PathInfo selectFromMinCostAndPath(vector<MatrixSection>& sections, vector<std::vector<int>>& originalMatrix, const pair<int, int>& rSource, const pair<int, int>& rSink, const vector<pair<int, int>>& uncPins)
    {
        int nSize = sections.size();
        int nSizePairDiag = (nSize+1)/2;
        //concatenate minCost and the path at each diag node
        vector<PathInfo> costFinal(nSizePairDiag);
        cout << "Eval final res process " << endl; 
        for(int i = 0; i < nSizePairDiag; i++)
        {
            if(sections[i].bPath.size() == 0 || sections[i + nSizePairDiag].bPath.size() == 0)
            {
                costFinal[i].cost = INF;
                continue;
            }
            costFinal[i].cost = sections[i].minCost + sections[i + nSizePairDiag].minCost - originalMatrix[sections[i + nSizePairDiag].bPath[0].first][sections[i + nSizePairDiag].bPath[0].second] + ( directionSame(sections[i].bPath, sections[i + nSizePairDiag].bPath) ? 0 : 50);
            if(directionSame(sections[i].bPath, sections[i + nSizePairDiag].bPath) == false)
            {
                sections[i].isLocVia[sections[i].bPath[sections[i].bPath.size() - 1]] = true;
            }
            // TODO @Jingren: Source in LU and Sink in RL, or set to INF. Corner case, also give a tie breaker.
            //if(!sections[i].hasLoc(rSource) || !sections[i + nSizePairDiag].hasLoc(rSink))
            //{
            //    costFinal[i].cost = INF;
            //}
            costFinal[i].path.reserve(sections[i].bPath.size() + sections[i + nSizePairDiag].bPath.size());
            costFinal[i].path.insert(costFinal[i].path.end(), sections[i].bPath.begin(), sections[i].bPath.end());
            costFinal[i].path.insert(costFinal[i].path.end(), sections[i + nSizePairDiag].bPath.begin() + 1, sections[i + nSizePairDiag].bPath.end());
            cout << "Original cost is " << costFinal[i].cost << endl;
            cout << "Original path is " << endl;
            for(auto iPathOri = 0 ; iPathOri < costFinal[i].path.size() ; iPathOri++)
            {
                cout << "(" << costFinal[i].path[iPathOri].first << "," << costFinal[i].path[iPathOri].second << ")";
            }
            cout << endl;
            //Update vias check inside the update
            costFinal[i].isLocVia.insert(sections[i].isLocVia.begin(), sections[i].isLocVia.end());
            costFinal[i].isLocVia.insert(sections[i + nSizePairDiag].isLocVia.begin(), sections[i + nSizePairDiag].isLocVia.end());
            costFinal[i].updateUncPins(originalMatrix, uncPins);
            cout << "Updated cost is " << costFinal[i].cost << endl;
            cout << "Updated path is " << endl;
            for(auto iPathNew = 0 ; iPathNew < costFinal[i].path.size() ; iPathNew++)
            {
                cout << "(" << costFinal[i].path[iPathNew].first << "," << costFinal[i].path[iPathNew].second << ")";
            }
            cout << endl;
        }

        // show all the costFinal
        std::sort(costFinal.begin(), costFinal.end(), 
                [](const PathInfo& a, const PathInfo& b) {
                    return a.cost < b.cost;
                });
        if(costFinal[0].cost == INF)
        {
            PathInfo nCreate = creatAInfoOnFly(originalMatrix, rSource, rSink);
            return nCreate;
        }
        if(costFinal[0].cost == INF)
        {
            cout << "Failed to find one. Probably not a corner case." << endl;
        }
        return costFinal[0];
    }
    
protected:
    /*!
    \brief Check the direction around the diag nodes.
    */
    bool directionSame(vector<pair<int, int>> &bPathLU, vector<pair<int, int>> &bPathRD)
    {
        static const int dx[] = {-1, 1, 0, 0};
        static const int dy[] = {0, 0, -1, 1};
        if(bPathLU.size() >= 2 && bPathRD.size() >= 2)
        {
            auto LUEnd = bPathLU[bPathLU.size() - 1];
            auto LUEndL = bPathLU[bPathLU.size() - 2];
            auto dirLU = pair<int, int>(LUEndL.first - LUEnd.first, LUEndL.second - LUEnd.second);
            auto RDStart = bPathRD[0];
            auto RDStartR = bPathRD[1];
            auto dirRD = pair<int, int>(RDStartR.first - RDStart.first, RDStartR.second - RDStart.second);
            if(dirLU == dirRD)
            {
                return true;
            }
            else {
                return false;
            }
        }
        return true;
    }
    /*
    Get the number of turns in the seq
    */
    int numTurns(const vector<pair<int, int >>& path)
    {
        int nTurn = 0;
        if(path.size() < 3)
        {
            return nTurn;
        }
        int i = 0;
        for(i = 1 ; i < path.size() - 1; i++)
        {
            auto dirPre = make_pair(path[i].first - path[i - 1].first, path[i].second - path[i - 1].second);
            auto dirNext = make_pair(path[i + 1].first - path[i].first, path[i+1].second - path[i].second);
            if(dirPre != dirNext)
            {
                nTurn++;
            }
        }
        return nTurn;
    }
    
    /*
    Update the real value of the cost and path
    */
    PathInfo realValCheck(std::vector<std::vector<int>>& originalMatrix, pair<int, int>& realSink, vector<pair<int, int>>&path, map<pair<int, int>, bool> isLocVia)
    {
        int i = 0;
        PathInfo realUpdatePathInfo;
        // Important! Might lead to wired value.
        realUpdatePathInfo.cost = 0;
        realUpdatePathInfo.path.clear();
        realUpdatePathInfo.isLocVia.clear();
        int nSize = originalMatrix.size();
        int &costUpdate = realUpdatePathInfo.cost;
        map<pair<int, int>, bool> &isViaUpdate = realUpdatePathInfo.isLocVia;
        vector<pair<int, int>> &pathUpdate = realUpdatePathInfo.path;
        // Only need to check for sink, because (0,0) must be on the path
        if(path.size() > 0 && (path[path.size() - 1].first + path[path.size() - 1].second == nSize - 1))
        {
            for( i = path.size() - 1; i >= 0; i--)
            {
                if(path[i].first == 0 && path[i].second == 0)
                {
                    costUpdate += originalMatrix[path[i].first][path[i].second];
                    break;
                }
                costUpdate += originalMatrix[path[i].first][path[i].second];
            }
            pathUpdate.insert(pathUpdate.end(), path.begin() + i, path.end() );
            for(auto item = path.begin() + i; item < path.end(); item++)
            {
                isViaUpdate[*item] = isLocVia[*item];
            }
        }
        else if(path.size() > 0 && (path[0].first + path[0].second == nSize - 1))
        {
            if(isLocVia.count(realSink) == 0)
            {
                realUpdatePathInfo.cost = INF;
                realUpdatePathInfo.path.clear();
                realUpdatePathInfo.isLocVia.clear();
                return realUpdatePathInfo;
            }
            for( i = 0 ; i < path.size(); i++)
            {
                if(path[i].first == realSink.first && path[i].second == realSink.second)
                {
                    costUpdate += originalMatrix[path[i].first][path[i].second];
                    break;
                }
                costUpdate += originalMatrix[path[i].first][path[i].second];
            }
            //i = (i == path.size()) ? path.size() - 1 : i;
            pathUpdate.insert(pathUpdate.end(), path.begin(), path.begin() + i + 1 );
            for(auto item = path.begin(); item < path.begin() + i + 1; item++)
            {
                isViaUpdate[*item] = isLocVia[*item];
                if(item == path.begin() + i)
                {
                    isViaUpdate[*item] = false;
                }
            }
        }
        cout << "Show path update " << endl;
        for(const auto & item: pathUpdate)
        {
            cout << "(" << item.first << "," << item.second << ") ";
        }
        cout << endl;
        cout << "Show the isVia map " << endl;
        for(const auto & item: isViaUpdate)
        {
            cout << "(" << item.first.first << "," << item.first.second << ") " << item.second << endl;
        }
        cout << endl;
        assert(pathUpdate.size() == isViaUpdate.size());
        assert(numTurns(pathUpdate) == realUpdatePathInfo.numVias());
        costUpdate += realUpdatePathInfo.numVias() * turnCost;
        return realUpdatePathInfo;
    }

private:
    vector<vector<int>>& originalMatrix;
    vector<MatrixSection>& sections;
    int numSections;

};


class DazeRouter {
    public:
        pair<int, int> route(const vector<vector<int>> &cost, const int N, const vector<pair<int, int>> &pins, vector<pair<int, int>> &res);
    
    protected:
        bool calBetweenTwoPins(const pair<int, int>& pinFirst, const pair<int, int>& pinSecond, const vector<vector<int>> &cost, const int N, vector<PathInfo>& allPathInfo);
        void swapOnCondition(pair<int, int>& pinLU, pair<int, int>& pinLR, vector<vector<int>>& costSwapOnCondition, const pair<int, int>& pinLUOri, const pair<int, int>& pinUROri, const vector<vector<int>>& cost, const int N);
        void preProcess(const pair<int, int> &source, const pair<int, int> &sink, const vector<vector<int>>& h_matrix, const int N);
        void getOriginalMatrix(const pair<int, int>& source, const vector<vector<int>> h_matrix, const int N);
        PathInfo findMinCostPath(const pair<int, int> &source, const pair<int, int> &sink, const vector<pair<int, int>> &uncPins);
        void updateRealPathMap(PathInfo& p, const pair<int, int>& rSource, const int& squareN, const bool& needFlip, int N);
        void cleanUpPath(vector<pair<int, int>> &res, const vector<PathInfo> &allPathInfo);
        bool calMinPathCost(const pair<int, int>& pinFirst, const pair<int, int>& pinSecond, const vector<vector<int>>& cost, int N, int& c, vector<pair<int, int>>& p);
    private:
        int squareN;
        std::vector<std::vector<int>> originalMatrix;
        std::vector<MatrixSection> sections;
};

inline pair<int, int> DazeRouter::route(const vector<vector<int>> &cost, const int N, const vector<pair<int, int>> &pins, vector<pair<int, int>> &res)
{
    vector<pair<int,int>> unCFinalPins;
    auto clocks = clock();
    auto computeClocks = clock();
    assert(pins.size() >= 2);
    vector<PathInfo> allPathInfo;
    unCFinalPins.clear();
    allPathInfo.clear();
    for(auto i = 1; i < pins.size(); i++)
    {
        calBetweenTwoPins(pins[i-1], pins[i], cost, N, allPathInfo);
    }
    //Since we do not encode connected attribute on the location, so we need an extra filtering on duplicated locations.
    cleanUpPath(res, allPathInfo);
    collectAllPinsOnPath(res, pins, unCFinalPins);
    
    if(unCFinalPins.size() != 0)
    {
        for(const auto &unPin : unCFinalPins)
        {
            // sort the pins based on distance relative to the current pin euclidean distance
            auto tmpSort = pins;
            sortByDistance(tmpSort, unPin);
            for(auto &pin : tmpSort)
            {
                if(find(unCFinalPins.begin(), unCFinalPins.end(), pin)!=unCFinalPins.end())
                {
                    continue;
                }
                if(pin != unPin)
                {
                    int c = 0;
                    vector<pair<int,int>> p;
                    p.clear();
                    //auto ret = calBetweenTwoPins(pin, unPin, cost, N, allPathInfo);
                    auto ret = calMinPathCost(pin, unPin, cost, N, c, p);
                    if(ret == true)
                    {
                        PathInfo fixP;
                        fixP.cons(c, p);
                        allPathInfo.push_back(fixP);
                        break;
                    }
                }
            }
        }
        unCFinalPins.clear();
        res.clear();
        cleanUpPath(res, allPathInfo);
        collectAllPinsOnPath(res, pins, unCFinalPins);
        assert(unCFinalPins.size() == 0);
    }
    computeClocks = clock() - computeClocks;
    clocks = clock() - clocks;
    return make_pair(clocks, computeClocks);
}

inline bool DazeRouter::calMinPathCost(const pair<int, int>& pinFirst, const pair<int, int>& pinSecond, 
    const vector<vector<int>>& cost, int N, int& c, vector<pair<int, int>>& p) {

    // Directions: up, down, left, right
    vector<pair<int, int>> directions = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};

    // Initialize distance and previous arrays
    vector<vector<int>> dist(N, vector<int>(N, INF));
    vector<vector<pair<int, int>>> prev(N, vector<pair<int, int>>(N, {-1, -1}));
    vector<vector<bool>> visited(N, vector<bool>(N, false));

    // Priority queue for Dijkstra's algorithm
    priority_queue<pair<int, pair<int, int>>, 
        vector<pair<int, pair<int, int>>>,
        greater<pair<int, pair<int, int>>>> pq;

    // Start from the first pin
    dist[pinFirst.first][pinFirst.second] = cost[pinFirst.first][pinFirst.second];
    pq.push({cost[pinFirst.first][pinFirst.second], pinFirst});

    while (!pq.empty()) {
    auto [currentCost, currentPos] = pq.top();
    pq.pop();

    int x = currentPos.first;
    int y = currentPos.second;

    if (visited[x][y]) continue;
    visited[x][y] = true;

    // If we reached the destination, we can break early
    if (currentPos == pinSecond) {
    break;
    }

    // Explore neighbors
    for (auto& dir : directions) {
    int nx = x + dir.first;
    int ny = y + dir.second;

    // Check if neighbor is within bounds
    if (nx >= 0 && nx < N && ny >= 0 && ny < N) {
    // Check if the cost is not INF (assuming INF is a very large number)
    if (cost[nx][ny] < INF) {
        int newCost = dist[x][y] + cost[nx][ny];
        
        if (newCost < dist[nx][ny]) {
            dist[nx][ny] = newCost;
            prev[nx][ny] = {x, y};
            pq.push({newCost, {nx, ny}});
        }
    }
    }
    }
    }

    // Check if destination is reachable
    if (dist[pinSecond.first][pinSecond.second] == INF) {
    return false; // No path found
    }

    // Reconstruct the path
    p.clear();
    pair<int, int> current = pinSecond;

    while (current != pinFirst) {
    p.push_back(current);
    current = prev[current.first][current.second];

    // Safety check to avoid infinite loop
    if (current == make_pair(-1, -1)) {
    return false;
    }
    }
    p.push_back(pinFirst);

    // Reverse to get path from start to end
    reverse(p.begin(), p.end());
    assert(p[0] == pinFirst && p[p.size() - 1] == pinSecond);
    //show the path
    cout << "THE path reconsidered " << endl;
    for(auto &loc: p)
    {
        cout << "(" << loc.first << "," << loc.second << ")";
    }
    cout << endl;

    c = dist[pinSecond.first][pinSecond.second];
    assert(c < INF);
    return true;
}

inline bool DazeRouter::calBetweenTwoPins(const pair<int, int>& pinFirst, const pair<int, int>& pinSecond, const vector<vector<int>> &cost, const int N, vector<PathInfo>& allPathInfo)
{
       cout << "+++++++++Working on (" << pinFirst.first << "," << pinFirst.second << ")(" <<pinSecond.first << "," << pinSecond.second<<")+++++++++" << endl;
        pair<int, int> pinLU;
        pair<int, int> pinLR;
        bool needFlip = false;
        
        pair<int, int> pinLUSwap;
        pair<int, int> pinLRSwap;
        if(pinFirst.first > pinSecond.first)
        {
            pinLUSwap = pinSecond;
            pinLRSwap = pinFirst;
        }
        else {
            pinLUSwap = pinFirst;
            pinLRSwap = pinSecond;
        }
        
        if(pinLRSwap.first >= pinLUSwap.first && pinLRSwap.second <= pinLUSwap.second)
        {
            cout << "Flipped" << endl;
            needFlip = true;
        }
        vector<vector<int>> costSwapOnCondition(N, vector<int> (N));
        squareN = abs(pinLRSwap.first - pinLUSwap.first) + 1 > abs(pinLRSwap.second - pinLUSwap.second) + 1 ? abs(pinLRSwap.first - pinLUSwap.first) + 1 : abs(pinLRSwap.second - pinLUSwap.second) + 1;
        cout << "SquareN is " << squareN << endl;
        swapOnCondition(pinLU, pinLR, costSwapOnCondition, pinLUSwap, pinLRSwap, cost, N);
        cout << "Left upper (" << pinLU.first << "," << pinLU.second << ") (" <<  pinLR.first << "," << pinLR.second << ")" <<endl;
        preProcess(pinLU, pinLR, costSwapOnCondition, N);
        //TODO @Jingren: Use special trival case: we iterate 2 nodes each time, we will iterate all, so no unconnected pins.
        vector<pair<int, int>> uncPins = {};
        PathInfo p = findMinCostPath(pinLU, pinLR, uncPins);
        if(p.cost == INF)
        {
            return false;
        }
        updateRealPathMap(p, pinLU, squareN, needFlip, N);
        allPathInfo.push_back(p);
        return true;
}

inline void DazeRouter::swapOnCondition(pair<int, int>& pinLU, pair<int, int>& pinLR, vector<vector<int>>& costSwapOnCondition, const pair<int, int>& pinLUOri, const pair<int, int>& pinUROri, const vector<vector<int>>& cost, const int N)
{
    assert(cost[pinLUOri.first][pinLUOri.second] != INF && cost[pinUROri.first][pinUROri.second] != INF);
    if(pinUROri.first >= pinLUOri.first && pinUROri.second <= pinLUOri.second)
    {
        assert(pinLUOri.first < pinUROri.first);
        //auto diff = abs(pinLUOri.second - pinUROri.second);
        //pinLU.first = pinLUOri.first;
        //pinLU.second = pinLUOri.second - diff;
        //pinLR.first = pinUROri.first;
        //pinLR.second = pinUROri.second + diff;
        pinLU.first = pinLUOri.first;
        pinLU.second = N - 1 - pinLUOri.second;
        pinLR.first = pinUROri.first;
        pinLR.second = N - 1-  pinUROri.second;
        
        cout << "swaped M" << endl;
        for(int i = 0; i < N; i++)
        {
            for(int j = 0; j < N; j++)
            {
                costSwapOnCondition[i][j] = cost[i][N - 1 - j];
                cout << costSwapOnCondition[i][j] << " ";
            }
            cout << endl;
        }
    }
    else {
        pinLU = pinLUOri;
        pinLR = pinUROri;
        for(int i = 0; i < N; i++)
        {
            for(int j = 0; j < N; j++)
            {
                costSwapOnCondition[i][j] = cost[i][j];
            }
        }
    }
}


inline void DazeRouter::updateRealPathMap(PathInfo& p, const pair<int, int>& rSource, const int& squareN, const bool& needFlip, int N)
{
    map<pair<int, int>, bool> tmpUpdate;
    for(auto &item:p.isLocVia)
    {
        int f,s;
        f = item.first.first + rSource.first;
        if(needFlip)
        {
            s = N - 1 - (item.first.second + rSource.second);
        }
        else {
            s = item.first.second + rSource.second;
        }
        auto itemUpdate = make_pair(f,s);
        tmpUpdate[itemUpdate] = item.second;
    }
    p.isLocVia.clear();
    p.isLocVia = tmpUpdate;
}

inline void DazeRouter::cleanUpPath(vector<pair<int, int>> &res, const vector<PathInfo> &allPathInfo)
{
    // Before cleanup, it is necessary that all real source should be added to the map(should be path and map, we only update map for simplicity.), if missing the update step, you will encounter having such as (0,0) in all PathInfo.
    set<pair<int, int>> tmpSaveUniqueLoc;
    for(const auto &pInfo : allPathInfo)
    {
        for(const auto [k, v] : pInfo.isLocVia)
        {
            tmpSaveUniqueLoc.insert(k);
        }
    }
    res = vector<pair<int, int>>(tmpSaveUniqueLoc.begin(), tmpSaveUniqueLoc.end());
}

inline void DazeRouter::getOriginalMatrix(const pair<int, int>& source, const vector<vector<int>> h_matrix, int N)
{
    originalMatrix.clear();
    assert(h_matrix[source.first][source.second] != INF);
    std::vector<std::vector<int>> originalMatrixExtract(squareN, std::vector<int>(squareN));
    for (int i = 0; i < squareN; i++) {
        for (int j = 0; j <squareN; j++) {
            // if exceed the boundaries, then set the cost to INF.
            //if(i + source.first >= N || ((i+source.first) * N + (j+source.second)) >= N)
            if(i + source.first >= N)
            {
                if(j == squareN - 1)
                {
                    originalMatrixExtract[i][j] = 0;
                }
                else {
                    originalMatrixExtract[i][j] = INF;
                }
            }
            else if(j + source.second >= N)
            {
                if(i == squareN - 1)
                {
                    originalMatrixExtract[i][j] = 0;
                }
                else {
                    originalMatrixExtract[i][j] = INF;
                }
            }
            else {
                //originalMatrixExtract[i][j] = h_matrix[i + source.first][(i+source.first) * N + (j+source.second)];
                originalMatrixExtract[i][j] = h_matrix[i + source.first][j+source.second];
            }
        }
    }
    assert(originalMatrixExtract[0][0] != INF);
    originalMatrix = originalMatrixExtract;
    cout << "Extracted matrix" << endl;
    for(auto i = 0; i < squareN; i++)
    {
        for(auto j = 0; j < squareN; j++)
        {
            cout << originalMatrix[i][j]<< " ";
        }
        cout<< endl;
    }
    cout << endl;
}

inline void DazeRouter::preProcess(const pair<int, int> &source, const pair<int, int> &sink, const vector<vector<int>>& h_matrix, const int N)
{
    sections.clear();
    const vector<pair<int, int>> indices = getSecDiagMatrixIndices(make_pair(0, 0), squareN);
    createSectionsOnSecDiag(indices, sections, make_pair(0, 0), make_pair(squareN - 1,  squareN - 1));
    getOriginalMatrix(source, h_matrix, N);
}

inline PathInfo DazeRouter::findMinCostPath(const pair<int, int> &source, const pair<int, int> &sink, const vector<pair<int, int>> &uncPins)
{
    MatrixSectionProcessorCPU processor(originalMatrix, sections);
    auto sinkCur = make_pair(sink.first - source.first, sink.second - source.second);
    auto sourceCur = make_pair(0, 0);
    processor.getMinCostAndPathOnSections(originalMatrix, sourceCur, sinkCur);
    PathInfo info = processor.selectFromMinCostAndPath(sections, originalMatrix, sourceCur, sinkCur, uncPins);
    return info;
}

#endif // SECTION_OPS_CPU_HPP