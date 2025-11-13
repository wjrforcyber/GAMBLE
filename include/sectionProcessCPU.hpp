#include <iostream>
#include <utility>
#include <vector>
#include "mazeRouter.hpp"

using namespace std;

#ifndef SECTION_OPS_CPU_HPP
#define SECTION_OPS_CPU_HPP

struct PathInfo {
    int cost;
    vector<pair<int, int>> path;
    void reverseInplace(int nSquare)
    {
        for(int i = 0 ; i < path.size(); i++)
        {
            path[i].second =  (nSquare - 1) - path[i].second;
        }
    }
    void costUncPin(std::vector<std::vector<int>>& originalMatrix, vector<pair<int, int>>& path, const pair<int, int>& pin)
    {
        vector<int> cost4Direction = {0,0,0,0};
        vector<pair<int, int>> extenUncDir = {pin, pin, pin, pin};
        //auto tmpPin = pin;
        static const int dx[] = {-1, 1, 0, 0};
        static const int dy[] = {0, 0, -1, 1};

        int iDir = 0;
        bool iFound = false;
        // It should be 100% sure that you could find the path to the original path according to the predefined scene
        //while(std::find(path.begin(), path.end(), extenUncDir[iDir]) == path.end())
        while(true)
        {
            for( iDir = 0; iDir < 4; iDir++)
            {
                if(extenUncDir[iDir].first + dx[iDir] >=0 && extenUncDir[iDir].first + dx[iDir] <= originalMatrix.size() - 1 \
                && extenUncDir[iDir].second + dy[iDir] >=0 && extenUncDir[iDir].second + dy[iDir] <= originalMatrix.size() - 1)
                {
                    extenUncDir[iDir].first += dx[iDir];
                    extenUncDir[iDir].second += dy[iDir];
                }
                else {
                    continue;
                }
                if (std::find(path.begin(), path.end(), extenUncDir[iDir]) != path.end()) {
                    std::cout << "Element found!" << std::endl;
                    iFound = true;
                    break;
                } else {
                    std::cout << "Element not found." << std::endl;
                }
            }
            if(iFound == true)
            {
                break;
            }
            //iDir = 0;
        }
        // fetch path from extenUncDir[iDir] to pin
        pair<int, int> tmpRecord = pin;
        int recordPreSize = path.size();
        while(tmpRecord != extenUncDir[iDir])
        {
            path.push_back(tmpRecord);
            cost += originalMatrix[tmpRecord.first][tmpRecord.second];
            tmpRecord.first += dx[iDir];
            tmpRecord.second += dy[iDir];
        }
        if(extenUncDir[iDir] != path[0] && extenUncDir[iDir] != path[recordPreSize - 1])
        {
            auto it = std::find(path.begin(), path.end(), extenUncDir[iDir]);
            int index = distance(path.begin(), it);
            //There's no via originally, now we bring ones
            if(path[index].first - path[index - 1].first == path[index + 1].first - path[index].first && \
            path[index].second - path[index - 1].second == path[index + 1].second - path[index].second)
            {
                cost += turnCost;
            }
            //If there's via, then at this point then we don't need to care.
            //TODO @Jingren This need to be changed to directly encode via at the location instead of checking!!!
        }
    }
    void updateUncPins(std::vector<std::vector<int>>& originalMatrix, const vector<pair<int, int>>& uncPins)
    {
        for(int i = 0 ;i < uncPins.size(); i++)
        {
            costUncPin(originalMatrix, path, uncPins[i]);
        }
    }
};

struct MatrixSection {
    int top, left;      // Upper-left corner
    int bottom, right;  // Lower-right corner
    vector< vector<int>> data; // Section data
    int rows, cols;     // Section dimensions
    int minCost;        // Mincost meet at the diag position
    vector<pair<int, int>> bPath; // Record all the path from (top, left) to (bottom, right)
    /*
    Check if the location is in the current path
    */
    bool hasLoc(pair<int, int> loc)
    {
        for(int i = 0; i < bPath.size(); i++)
        {
            if(bPath[i] == loc)
            {
                return true;
            }
        }
        return false;
    }
    
    /*
        calculate the shortest path sum from top-left to bottom-right, each position has cost equal to the value at that position in data, moving left, right, up, down only, however, each turn costs an additional fixed cost (turnCost).
    */
    PathInfo shortestPathSumWithTurnCost() {
        const int INF = 1e9;
        
        // dp[r][c][d] = min cost to reach (r,c) facing direction d
        vector<vector<vector<int>>> dp(rows, 
            vector<vector<int>>(cols, 
                vector<int>(4, INF)));
        
        // Store the entire path for each state
        vector<vector<vector<vector<pair<int, int>>>>> paths(rows,
            vector<vector<vector<pair<int, int>>>>(cols,
                vector<vector<pair<int, int>>>(4)));
        
        using Node = tuple<int, int, int, int>;
        priority_queue<Node, vector<Node>, greater<Node>> pq;
        
        // Initialize starting point
        dp[0][0][1] = data[0][0];
        dp[0][0][3] = data[0][0];
        paths[0][0][1] = {{0, 0}};
        paths[0][0][3] = {{0, 0}};
        pq.push({data[0][0], 0, 0, 1});
        pq.push({data[0][0], 0, 0, 3});
        
        vector<pair<int, int>> dirs = {{-1,0}, {1,0}, {0,-1}, {0,1}};
        
        PathInfo result;
        result.cost = INF;
        
        while (!pq.empty()) {
            auto [cost, r, c, dir] = pq.top();
            pq.pop();
            
            if (cost > dp[r][c][dir]) continue;
            
            // If we reached destination, check if this is the best path
            if (r == rows-1 && c == cols-1) {
                if (cost < result.cost) {
                    result.cost = cost;
                    result.path = paths[r][c][dir];
                }
            }
            
            for (int newDir = 0; newDir < 4; newDir++) {
                int nr = r + dirs[newDir].first;
                int nc = c + dirs[newDir].second;
                
                if (nr < 0 || nr >= rows || nc < 0 || nc >= cols) continue;
                
                int newCost = cost + data[nr][nc];
                if (dir != newDir) {
                    newCost += turnCost;
                }
                
                if (newCost < dp[nr][nc][newDir]) {
                    dp[nr][nc][newDir] = newCost;
                    
                    // Update path
                    paths[nr][nc][newDir] = paths[r][c][dir];
                    paths[nr][nc][newDir].push_back({nr, nc});
                    
                    pq.push({newCost, nr, nc, newDir});
                }
            }
        }
        // Update path with offset
        for(auto i = 0 ; i < result.path.size(); i++)
        {
            result.path[i].first += top;
            result.path[i].second += left;
        }
        return result;
    }
    
    bool hasTargetInSection(pair<int, int> targetNode)
    {
        if(targetNode.first >= top && targetNode.second >= left && targetNode.first <= bottom && targetNode.second <= right)
            return true;
        return false;
    }
};


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
            auto [minCost, path] = sections[i].shortestPathSumWithTurnCost();
            cout << "Original Path Sum with Turn Cost: " << minCost << endl;
            //show the path
            cout << "Path: ";
            for (auto [r, c] : path) {
                cout << "(" << r << "," << c << ") ";
            }
            cout << endl;
            // TODO: @Jingren Wang: get the real value from diag to source and from diag to sink
            auto [minCostUpdate, pathUpdate] = realValCheck(originalMatrix, realSource, realSink, path);

            cout << "Shortest Path Sum with Turn Cost: " << minCostUpdate << endl;
            //show the path
            cout << "Path: ";
            for (auto [r, c] : pathUpdate) {
                cout << "(" << r << "," << c << ") ";
            }
            // pass the minCost and path to section[i]
            sections[i].minCost = minCostUpdate;
            sections[i].bPath = pathUpdate;
            cout << endl;
            cout << endl;
        }
    }
    
    //get sections
    vector<MatrixSection>& getSections() {
        return sections;
    }
    
protected:
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
    PathInfo realValCheck(std::vector<std::vector<int>>& originalMatrix, pair<int, int>& realSource, pair<int, int>& realSink, vector<pair<int, int>>&path)
    {
        //int realUpdate = 0;
        //vector<pair<int, int>> pathUpdate;
        //PathInfo realUpdatePathInfo;
        //int nSize = originalMatrix.size();
        //int i = 0;
        //if(path.size() > 0 && (path[path.size() - 1].first + path[path.size() - 1].second == nSize - 1))
        //{
        //    for(i = path.size() - 1; i > 0; i--)
        //    {
        //        realUpdate += originalMatrix[path[i].first][path[i].second];
        //        if(path[i].first == realSource.first && path[i].second == realSource.second)
        //            break;
        //    }
        //    pathUpdate.insert( pathUpdate.end(), path.begin() , path.end());
        //}
        //else if(path.size() > 0 && (path[0].first + path[0].second == nSize - 1)){
        //    for(i = 0; i < path.size(); i++)
        //    {
        //        realUpdate += originalMatrix[path[i].first][path[i].second];
        //        if(path[i].first == realSink.first && path[i].second == realSink.second)
        //            break;
        //    }
        //    pathUpdate.insert( pathUpdate.end(), path.begin(), path.begin() + i);
        //}
        //else {
        //    cout << "Do not encounter with the condition." << endl;
        //}
        //cout << "The updated cost is " << realUpdate << endl;
        //realUpdatePathInfo.cost = realUpdate;
        //realUpdatePathInfo.path = pathUpdate;
        //return realUpdatePathInfo;
        int i = 0;
        int costUpdate = 0;
        PathInfo realUpdatePathInfo;
        int nSize = originalMatrix.size();
        vector<pair<int, int>> pathUpdate;
        if(path.size() > 0 && (path[path.size() - 1].first + path[path.size() - 1].second == nSize - 1))
        {
            for( i = path.size() - 1; i >= 0; i--)
            {
                if(path[i].first == realSource.first && path[i].second == realSource.second)
                {
                    costUpdate += originalMatrix[path[i].first][path[i].second];
                    break;
                }
                costUpdate += originalMatrix[path[i].first][path[i].second];
            }
            pathUpdate.insert(pathUpdate.end(), path.begin() + i, path.end() );
        }
        else if(path.size() > 0 && (path[0].first + path[0].second == nSize - 1))
        {
            for( i = 0 ; i < path.size(); i++)
            {
                if(path[i].first == realSink.first && path[i].second == realSink.second)
                {
                    costUpdate += originalMatrix[path[i].first][path[i].second];
                    break;
                }
                costUpdate += originalMatrix[path[i].first][path[i].second];
            }
            i = (i == path.size()) ? path.size() - 1 : i;
            pathUpdate.insert(pathUpdate.end(), path.begin(), path.begin() + i + 1 );
        }
        realUpdatePathInfo.cost = costUpdate + numTurns(pathUpdate) * turnCost;
        realUpdatePathInfo.path = pathUpdate;
        return realUpdatePathInfo;
    }

private:
    vector<vector<int>>& originalMatrix;
    vector<MatrixSection>& sections;
    int numSections;

};
#endif // SECTION_OPS_CPU_HPP