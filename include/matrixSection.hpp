#ifndef SECTION_MATRIX_CPU_HPP
#define SECTION_MATRIX_CPU_HPP

#include <cassert>
#include <iostream>
#include <utility>
#include <vector>
#include "mazeRouter.hpp"
#include "pathInfo.hpp"
struct MatrixSection {
    int top, left;      // Upper-left corner
    int bottom, right;  // Lower-right corner
    vector< vector<int>> data; // Section data
    int rows, cols;     // Section dimensions
    int minCost;        // Mincost meet at the diag position
    vector<pair<int, int>> bPath; // Record all the path from (top, left) to (bottom, right)
    map<pair<int, int>, bool>  isLocVia;
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
        result.path.clear();
        result.isLocVia.clear();
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
        for(auto i = 0 ; i < result.path.size(); i++)
        {
            // check the final path decision, if it is a via point
            if(i == 0 || i == result.path.size() - 1)
            {
                result.isLocVia[result.path[i]] = false;
            }
            if(i != 0 && i != result.path.size() - 1)
            {
                if((result.path[i].first - result.path[i - 1].first == result.path[i+1].first - result.path[i].first) && \
                   (result.path[i].second - result.path[i - 1].second == result.path[i+1].second - result.path[i].second) )
                {
                    result.isLocVia[result.path[i]] = false;
                }
                else {
                    result.isLocVia[result.path[i]] = true;
                }
            }
        }
        assert(result.isLocVia.size() == result.path.size());
        return result;
    }
    
    bool hasTargetInSection(pair<int, int> targetNode)
    {
        if(targetNode.first >= top && targetNode.second >= left && targetNode.first <= bottom && targetNode.second <= right)
            return true;
        return false;
    }
};

#endif