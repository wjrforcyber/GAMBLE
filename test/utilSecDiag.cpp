#include "util.hpp"
#include <cassert>
#include <utility>

int main() {
    int squareN;
    // The indices must has real source and sink as para
    pair<int,int> source = {0,0};
    pair<int,int> sink = {4,4};
    squareN = abs(source.first - sink.first) + 1 > abs(source.second - sink.second) + 1 ? abs(source.first - sink.first) + 1 : abs(source.second - sink.second) + 1;
    auto indices = getSecDiagMatrixIndices(make_pair(0, 0), squareN);
    assert(squareN == 5);
    // Expected secondary diagonal indices for a 5x5 matrix
    std::vector<std::pair<int, int>> expected = {
        {0, 4},
        {1, 3}, 
        {2, 2}, 
        {3, 1}, 
        {4, 0}
    };
    assert(indices.size() == expected.size());
    for (size_t i = 0; i < expected.size(); i++) {
        assert(indices[i] == expected[i]);
    }
    
    return 0;
}