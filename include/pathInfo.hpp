#ifndef PATH_INFO_CPU_HPP
#define PATH_INFO_CPU_HPP

#include <vector>
#include <iostream>
#include <map>
#include <cassert>
#include "mazeRouter.hpp"
using namespace std;
struct PathInfo {
    int cost;
    vector<pair<int, int>> path;
    map<pair<int,int>, bool> isLocVia;
    // check if a location is via
    bool isVia(pair<int, int> loc)
    {
        if(isLocVia.size() == 0)
        {
            cout << "Error: Location of the via is not properly decided." << endl;
            return false;
        }
        if(isLocVia.count(loc) == 0)
        {
            cout << "Error: Position does not exit." << endl;
            return false;
        }
        return isLocVia[loc];
    }
    
    // number of vias on path
    int numVias()
    {
        int count = 0;
        for(auto &item : isLocVia)
        {
            if(isVia(item.first))
            {
                count++;
            }
        }
        return count;
    }

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
        }
        // fetch path from extenUncDir[iDir] to pin
        pair<int, int> tmpRecord = pin;
        int recordPreSize = path.size();
        while(tmpRecord != extenUncDir[iDir])
        {
            path.push_back(tmpRecord);
            assert(isLocVia.count(tmpRecord) == 0);
            isLocVia[tmpRecord] = false;
            cost += originalMatrix[tmpRecord.first][tmpRecord.second];
            tmpRecord.first += dx[iDir];
            tmpRecord.second += dy[iDir];
        }
        // if the attached point is not the source and sink point, we could directly check
        if(extenUncDir[iDir] != path[0] && extenUncDir[iDir] != path[recordPreSize - 1])
        {
            auto it = std::find(path.begin(), path.end(), extenUncDir[iDir]);
            int index = distance(path.begin(), it);
            //There's no via originally, now we bring ones
            if(isVia(path[index]) == false)
            {
                // It's now a via, update
                isLocVia[path[index]] = true;
                cost += turnCost;
            }
        }
        // if the attached point is at the beginning or the end check the ini dir with iDir, or end dir with iDir
        if(extenUncDir[iDir] == path[0])
        {
            auto tmpDir = make_pair(dx[iDir], dy[iDir]);
            if(tmpDir != make_pair(path[1].first - path[0].first, path[1].second - path[0].second))
            {
                assert(isLocVia[path[0]] == false);
                isLocVia[path[0]] = true;
                cost += turnCost;
            }
        }
        if(extenUncDir[iDir] == path[recordPreSize - 1])
        {
            auto tmpDir = make_pair(dx[iDir], dy[iDir]);
            if(tmpDir != make_pair(path[recordPreSize - 1].first - path[recordPreSize - 2].first, path[recordPreSize - 1].second - path[recordPreSize - 2].second))
            {
                assert(isLocVia[path[recordPreSize - 1]] == false);
                isLocVia[path[recordPreSize - 1]] = true;
                cost += turnCost;
            }
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

#endif
