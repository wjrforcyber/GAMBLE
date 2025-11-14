#include<vector>
#include "sectionProcessCPU.hpp"

using namespace std;

#ifndef UTILS_HPP
#define UTILS_HPP


struct Matrix
{
    vector<vector<int>> m;
    //saved for other features
};

vector<pair<int, int>> getSecDiagMatrixIndices(pair<int, int> LU, pair<int,int> RD, const int& squareN);
void createSectionsOnSecDiag(const vector<pair<int, int>>& indices, vector<MatrixSection>& sections, pair<int, int> LU, pair<int, int> BR);
bool directionSame(vector<pair<int, int>> &bPathLU, vector<pair<int, int>> &bPathRD);
PathInfo selectFromMinCostAndPath(vector<MatrixSection>& sections, std::vector<std::vector<int>>& originalMatrix, pair<int, int>& rSource, pair<int, int>& rSink, vector<pair<int, int>>& uncPins);
vector<Matrix> partLargeM(const vector<vector<int>>& inM, const int nWin);


#endif // UTILS_HPP