#include <algorithm>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/extrema.h>
#include <thrust/functional.h>


/*!
    \brief Normalize the matrix by subtracting the minimum value from all elements on CPU
*/
void normalizeMatrixCPU(std::vector<float>& matrix) {
    float min_val = *std::min_element(matrix.begin(), matrix.end());
    for (auto& val : matrix) {
        val -= min_val;
    }
}

/*!
    \brief Normalize the matrix by subtracting the minimum value from all elements using Thrust library on GPU
*/
void normalizeMatrixThrust(thrust::device_vector<float>& d_matrix) {
    // Find minimum value
    auto min_iter = thrust::min_element(d_matrix.begin(), d_matrix.end());
    float min_val = *min_iter;
    
    // Subtract minimum from all elements
    thrust::transform(d_matrix.begin(), d_matrix.end(), d_matrix.begin(),
                     [min_val] __device__ (float x) { return x - min_val; });
}

int main()
{
    // Give a matrix that has 4 * 4 elements and each element is between 51 to 60
    std::vector<float> h_matrix = {
        51.0f, 52.0f, 53.0f, 54.0f,
        55.0f, 56.0f, 57.0f, 58.0f,
        59.0f, 60.0f, 51.0f, 52.0f,
        53.0f, 54.0f, 55.0f, 56.0f
    };
    // Give a matrix that has 4 * 4 elements and each element is between 51 to 60
    std::vector<float> h_matrix_cpu = {
        51.0f, 52.0f, 53.0f, 54.0f,
        55.0f, 56.0f, 57.0f, 58.0f,
        59.0f, 60.0f, 51.0f, 52.0f,
        53.0f, 54.0f, 55.0f, 56.0f
    };
    
    // Give the normalized result
    std::vector<float> r_matrix = {
        0.0f, 1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f, 7.0f,
        8.0f, 9.0f, 0.0f, 1.0f,
        2.0f, 3.0f, 4.0f, 5.0f
    };
    
    auto clocksGPUs = clock();
    thrust::device_vector<float> d_matrix = h_matrix;
    normalizeMatrixThrust(d_matrix);
    // Copy back to host and print
    thrust::copy(d_matrix.begin(), d_matrix.end(), h_matrix.begin());
    auto clocksGPUDiff = clock() - clocksGPUs;
    // Print the GPU time cost
    std::cout << "GPU Time cost: " << ((float)clocksGPUDiff) / CLOCKS_PER_SEC << " seconds." << std::endl;
    // assert two matrix are same
    for (size_t i = 0; i < h_matrix.size(); i++) {
        assert(h_matrix[i] == r_matrix[i]);
    }
    
    auto clocksCPUs = clock();
    normalizeMatrixCPU(h_matrix_cpu);
    auto clocksCPUDiff = clock() - clocksCPUs;
    //print the CPU time cost
    std::cout << "CPU Time cost: " << ((float)clocksCPUDiff) / CLOCKS_PER_SEC << " seconds." << std::endl;
    // assert two matrix are same
    for (size_t i = 0; i < h_matrix_cpu.size(); i++) {
        assert(h_matrix_cpu[i] == r_matrix[i]);
    }
    
    // initialize a matrix with 10000 * 10000 elements
    const int N = 1000;
    std::vector<float> large_matrix(N * N);
    // Fill the matrix with random values between 0 to 1000
    for (int i = 0; i < N * N; i++) {
        large_matrix[i] = static_cast<float>(rand() % 1000);
    }
    std::vector<float> large_matrix_cpu = large_matrix;
    thrust::device_vector<float> d_large_matrix = large_matrix;
    // Normalize using GPU
    auto clocksLargeGPUs = clock();
    normalizeMatrixThrust(d_large_matrix);
    auto clocksLargeGPUDiff = clock() - clocksLargeGPUs;
    std::cout << "Large GPU Time cost: " << ((float)clocksLargeGPUDiff) / CLOCKS_PER_SEC << " seconds." << std::endl;
    // Normalize using CPU
    auto clocksLargeCPUs = clock();
    normalizeMatrixCPU(large_matrix_cpu);
    auto clocksLargeCPUDiff = clock() - clocksLargeCPUs;
    std::cout << "Large CPU Time cost: " << ((float)clocksLargeCPUDiff) / CLOCKS_PER_SEC << " seconds." << std::endl;
    return 0;
}