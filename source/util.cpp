#include "util.hpp"
#include <utility>
#include <vector>
/*!
  \brief Create second diag position.
*/
vector<pair<int, int>> getSecDiagMatrixIndices(pair<int, int> LU, const int& squareN) {
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

/*!
  \brief Helper function on checking all pins on the result path.
*/
bool checkAllPinsOnPath(const vector<pair<int, int>>& res, const vector<pair<int, int>>& pins)
{
    for(const auto &item: pins)
    {
        auto it = std::find(res.begin(), res.end(), item);
        if(it != res.end())
        {
            continue;
        }
        else {
            return false;
        }
    }
    return true;
}