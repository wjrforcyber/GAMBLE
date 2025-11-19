
#ifndef UTILS_HPP
#define UTILS_HPP
#include<vector>
#include "matrixSectionGPU.hpp"
using namespace std;

struct Matrix
{
    vector<vector<int>> m;
    //saved for other features
};
struct MatrixSection;
vector<pair<int, int>> getSecDiagMatrixIndices(pair<int, int> LU, const int& squareN);
void createSectionsOnSecDiag(const vector<pair<int, int>>& indices, vector<MatrixSection>& sections, pair<int, int> LU, pair<int, int> BR);
vector<Matrix> partLargeM(const vector<vector<int>>& inM, const int nWin);
bool checkAllPinsOnPath(const vector<pair<int, int>>& res, const vector<pair<int, int>>& pins);
void collectAllPinsOnPath(const vector<pair<int, int>>& res, const vector<pair<int, int>>& pins, vector<pair<int, int>>& unCFinalPins);


#endif // UTILS_HPP