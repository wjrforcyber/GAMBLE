#include <cassert>
#include <vector>
#include <iostream>
#include "util.hpp"

using namespace std;

int main() {
    // Create 13x13 test matrix
    vector<vector<int>> matrix(13, vector<int>(13));
    int counter = 1;
    cout << "Show init matrix" << endl;
    for (int i = 0; i < 13; i++) {
        for (int j = 0; j < 13; j++) {
            matrix[i][j] = counter++;
            cout << matrix[i][j] << " ";
        }
        cout << endl;
    }

    // Create 14x14 test matrix
    vector<vector<int>> matrixA(14, vector<int>(14));
    counter = 1;
    for (int i = 0; i < 14; i++) {
        for (int j = 0; j < 14; j++) {
            matrixA[i][j] = counter++;
        }
    }
    // Partition the matrix
    auto partitions = partLargeM(matrix, 4);
    assert(partitions.size() == 16);
    auto partitionsA = partLargeM(matrixA, 4);
    assert(partitionsA.size() == 25);

    for(auto &mEach: partitions)
    {
        for(int row = 0 ; row < mEach.m.size(); row++)
        {
            for(int col = 0; col < mEach.m[0].size(); col++)
            {
                cout << mEach.m[row][col] << " ";
            }
            cout << endl;
        }
        cout << endl;
    }
    return 0;
}