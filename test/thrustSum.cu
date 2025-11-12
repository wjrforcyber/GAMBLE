
#include <thrust/device_vector.h>
#include <thrust/reduce.h>

/*!
  \brief Sum up the matrix using Thrust library on GPU
*/
int sumMatrix(const std::vector<std::vector<int>>& matrix) {
    // Flatten the 2D vector to 1D
    std::vector<int> flattened;
    for (const auto& row : matrix) {
        flattened.insert(flattened.end(), row.begin(), row.end());
    }
    
    // Copy to device and sum
    thrust::device_vector<int> d_vec = flattened;
    return thrust::reduce(d_vec.begin(), d_vec.end());
}

int main()
{
    // Example usage of sumMatrix
    std::vector<std::vector<int>> matrix = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    };
    int totalSum = sumMatrix(matrix);
    assert(totalSum == 45); // 1+2+3+4+5+6+7+8+9 = 45
    return 0;
}
