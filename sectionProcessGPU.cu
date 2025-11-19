#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include "sectionProcessGPU.hpp"

// Kernel to copy sections from source matrix
__global__ void copySectionsKernel(int* source, int sourceRows, int sourceCols,
    int** sections, MatrixSection* sectionInfo, int numSections) {
    
    for (int sectionId = 0; sectionId < numSections; sectionId++) {
        MatrixSection section = sectionInfo[sectionId];
        int* sectionData = sections[sectionId];
        
        for (int row = 0; row < section.rows; row++) {
            for (int col = 0; col < section.cols; col++) {
                int sourceRow = section.top + row;
                int sourceCol = section.left + col;
                
                if (sourceRow < sourceRows && sourceCol < sourceCols) {
                    sectionData[row * section.cols + col] = source[sourceRow * sourceCols + sourceCol];
                }
            }
        }
    }
}

// Kernel to process sections (example: multiply each element by 2)
__global__ void processSectionsKernel(int** sections, MatrixSection* sectionInfo, 
       int numSections, int* results) {
    
    for (int sectionId = 0; sectionId < numSections; sectionId++) {
        MatrixSection section = sectionInfo[sectionId];
        int* sectionData = sections[sectionId];
        
        //TODO @Jingren Wang: Could processing the current section here.
    }
}