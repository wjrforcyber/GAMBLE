#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include "sectionProcessGPU.hpp"

// Kernel to copy sections from source matrix
__global__ void copySectionsKernel(int* source, int sourceRows, int sourceCols,
                                  int** sections, MatrixSection* sectionInfo, 
                                  int numSections) {
    int sectionId = blockIdx.z;
    if (sectionId >= numSections) return;
    
    MatrixSection info = sectionInfo[sectionId];
    int* sectionData = sections[sectionId];
    
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row < info.rows && col < info.cols) {
        int sourceRow = info.top + row;
        int sourceCol = info.left + col;
        int sourceIdx = sourceRow * sourceCols + sourceCol;
        int sectionIdx = row * info.cols + col;
        
        sectionData[sectionIdx] = source[sourceIdx];
    }
}

// Kernel to process each section (example processing)
__global__ void processSectionsKernel(int** sections, MatrixSection* sectionInfo, 
                                     int numSections, int* results) {
    int sectionId = blockIdx.z;
    if (sectionId >= numSections) return;
    
    MatrixSection info = sectionInfo[sectionId];
    int* sectionData = sections[sectionId];
    
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row < info.rows && col < info.cols) {
        int idx = row * info.cols + col;
        // Example processing: square each element
        sectionData[idx] = sectionData[idx] * sectionData[idx];
        
        // Optional: reduction for result per section
        if (row == 0 && col == 0) {
            results[sectionId] = 0.0f; // Initialize
        }
        __syncthreads();
        
        // Simple atomic add for demonstration
        atomicAdd(&results[sectionId], sectionData[idx]);
    }
}

// Kernel to copy processed sections back to original matrix
__global__ void copyBackSectionsKernel(int* dest, int destRows, int destCols,
                                      int** sections, MatrixSection* sectionInfo,
                                      int numSections) {
    int sectionId = blockIdx.z;
    if (sectionId >= numSections) return;
    
    MatrixSection info = sectionInfo[sectionId];
    int* sectionData = sections[sectionId];
    
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row < info.rows && col < info.cols) {
        int destRow = info.top + row;
        int destCol = info.left + col;
        int destIdx = destRow * destCols + col;
        int sectionIdx = row * info.cols + col;
        
        dest[destIdx] = sectionData[sectionIdx];
    }
}