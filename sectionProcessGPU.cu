#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include "sectionProcessGPU.hpp"

// Kernel to copy sections from source matrix

__global__ void copySectionsKernel(int* source, int sourceRows, int sourceCols,
                                  int** sections, MatrixSection* sectionInfo, int numSections) {
    
    //printf("Kernel: copySectionsKernel launched with %d sections\n", numSections);
    
    for (int sectionId = 0; sectionId < numSections; sectionId++) {
        // Get section info from device memory (don't make local copy)
        MatrixSection* section = &sectionInfo[sectionId];
        int* sectionData = sections[sectionId];  // This is the correct device pointer
        
        //printf("Kernel: Processing section %d: %dx%d at (%d,%d)\n", 
        //       sectionId, section->rows, section->cols, section->top, section->left);
        
        for (int row = 0; row < section->rows; row++) {
            for (int col = 0; col < section->cols; col++) {
                int sourceRow = section->top + row;
                int sourceCol = section->left + col;
                
                if (sourceRow < sourceRows && sourceCol < sourceCols) {
                    int sourceIdx = sourceRow * sourceCols + sourceCol;
                    int sectionIdx = row * section->cols + col;
                    
                    // Copy directly from source to section device memory
                    sectionData[sectionIdx] = source[sourceIdx];
                    
                    // Debug: print first few assignments
                    if (sectionId == 0 && row < 2 && col < 2) {
                        //printf("Kernel: section[%d][%d] = source[%d][%d] = %d\n", 
                            //   row, col, sourceRow, sourceCol, source[sourceIdx]);
                    }
                } else {
                    // Handle out-of-bounds
                    sectionData[row * section->cols + col] = 0;
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