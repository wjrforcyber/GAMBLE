#include "util.hpp"
#include <utility>
#include <vector>
/*!
  \brief Create second diag position.
*/
vector<pair<int, int>> getSecDiagMatrixIndices(pair<int, int> LU, pair<int,int> RD, const int& squareN) {
    vector<pair<int, int>> indices;
    for (int r = 0; r < squareN; r++) {
        int c = squareN - 1 - r;
        if (c >= 0 && c < squareN) {
            indices.push_back({r + LU.first, c + LU.second});
        }
    }
    return indices;
}

/*
create sections based on secondary diagonal indices as (bottom-right) while (top-left) corners are (0,0) and (rows-1,cols-1)
*/
void createSectionsOnSecDiag(const vector<pair<int, int>>& indices, vector<MatrixSection>& sections, pair<int, int> LU, pair<int, int> BR)
{
    for (const auto& idx : indices) {
        int r = idx.first;
        int c = idx.second;
        sections.push_back({LU.first, LU.second, r, c});
    }
    for (const auto& idx : indices) {
        int r = idx.first;
        int c = idx.second;
        sections.push_back({r, c, BR.first, BR.second});
    }
}

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

PathInfo selectFromMinCostAndPath(vector<MatrixSection>& sections, std::vector<std::vector<int>>& originalMatrix, pair<int, int>& rSource, pair<int, int>& rSink, vector<pair<int, int>>& uncPins)
{
    int nSize = sections.size();
    int nSizePairDiag = (nSize+1)/2;
    //concatenate minCost and the path at each diag node
    vector<PathInfo> costFinal(nSizePairDiag);
    cout << "Eval final res process " << endl; 
    for(int i = 0; i < nSizePairDiag; i++)
    {
        costFinal[i].cost = sections[i].minCost + sections[i + nSizePairDiag].minCost - originalMatrix[sections[i + nSizePairDiag].bPath[0].first][sections[i + nSizePairDiag].bPath[0].second] + ( directionSame(sections[i].bPath, sections[i + nSizePairDiag].bPath) ? 0 : 50);
        if(directionSame(sections[i].bPath, sections[i + nSizePairDiag].bPath) == false)
        {
            sections[i].isLocVia[sections[i].bPath[sections[i].bPath.size() - 1]] = true;
        }
        // TODO @Jingren: Source in LU and Sink in RL, or set to INF. Corner case, also give a tie breaker.
        if(!sections[i].hasLoc(rSource) || !sections[i + nSizePairDiag].hasLoc(rSink))
        {
            costFinal[i].cost = INF;
        }
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
        // This will update both cost and paths
        if(costFinal[i].cost == INF)
        {
            cout << "   Cost defined as large, continue." << endl;
            continue;
        }
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
        cout << "Failed to find one. Probably not a corner case." << endl;
    }
    return costFinal[0];
}


/*!
  \brief Create window on fixed size of square. Perform previous method only inside of each window.
  This won't get a connected graph, between windows, there's high possibility that there's no connection.
*/

vector<Matrix> partLargeM(const vector<vector<int>>& inM, const int nWin)
{
    int rows = inM.size();
    int cols = inM[0].size();
    
    int step = nWin - 1;
    int numRowParts = ceil((rows - nWin) / static_cast<double>(step)) + 1;
    int numColParts = ceil((cols - nWin) / static_cast<double>(step)) + 1;
    int totalPartitions = numRowParts * numColParts;
    vector<Matrix> results;
    results.reserve(totalPartitions);

    // Generate partitions
    for (int i = 0; i < numRowParts; i++) {
        for (int j = 0; j < numColParts; j++) {
            int startRow, startCol;
            // Calculate starting positions (ensuring complete last partitions)
            if (i == numRowParts - 1) {
                startRow = rows - nWin;
            } else {
                startRow = i * step;
            }
            if (j == numColParts - 1) {
                startCol = cols - nWin;
            } else {
                startCol = j * step;
            }
            Matrix subMatrix;
            subMatrix.m = vector<vector<int>>(nWin, vector<int>(nWin));
            // Extract submatrix into the current partition
            for (int x = 0; x < nWin; x++) {
                for (int y = 0; y < nWin; y++) {
                    subMatrix.m[x][y] = inM[startRow + x][startCol + y];
                }
            }
            results.push_back(subMatrix);
        }
    }
    return results;
}