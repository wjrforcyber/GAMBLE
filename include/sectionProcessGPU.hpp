#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#ifndef SECTION_OPS_HPP
#define SECTION_OPS_HPP

struct MatrixSection {
    int top, left;      // Upper-left corner
    int bottom, right;  // Lower-right corner
    int* d_data;      // Device pointer for this section
    int rows, cols;     // Section dimensions
};

__global__ void copySectionsKernel(int* source, int sourceRows, int sourceCols,
    int** sections, MatrixSection* sectionInfo, 
    int numSections);
__global__ void processSectionsKernel(int** sections, MatrixSection* sectionInfo, 
    int numSections, int* results);
__global__ void copyBackSectionsKernel(int* dest, int destRows, int destCols,
    int** sections, MatrixSection* sectionInfo,
    int numSections);


class MatrixSectionProcessor {
private:
    int** d_sections;
    MatrixSection* d_sectionInfo;
    int numSections;
    
public:
    MatrixSectionProcessor(const std::vector<MatrixSection>& sections) {
        numSections = sections.size();
        
        // Allocate device memory for section pointers and info
        cudaMalloc(&d_sections, numSections * sizeof(int*));
        cudaMalloc(&d_sectionInfo, numSections * sizeof(MatrixSection));
        
        // Copy section info to device
        std::vector<MatrixSection> h_sectionInfo = sections;
        for (int i = 0; i < numSections; i++) {
            h_sectionInfo[i].rows = sections[i].bottom - sections[i].top;
            h_sectionInfo[i].cols = sections[i].right - sections[i].left;
            
            // Allocate device memory for each section
            size_t sectionSize = h_sectionInfo[i].rows * h_sectionInfo[i].cols * sizeof(int);
            cudaMalloc(&h_sectionInfo[i].d_data, sectionSize);
        }
        
        cudaMemcpy(d_sectionInfo, h_sectionInfo.data(), 
                  numSections * sizeof(MatrixSection), cudaMemcpyHostToDevice);
        
        // Copy section pointers array
        std::vector<int*> h_sectionPtrs(numSections);
        for (int i = 0; i < numSections; i++) {
            h_sectionPtrs[i] = h_sectionInfo[i].d_data;
        }
        cudaMemcpy(d_sections, h_sectionPtrs.data(), 
                  numSections * sizeof(int*), cudaMemcpyHostToDevice);
    }
    
    ~MatrixSectionProcessor() {
        // Free section device memory
        std::vector<MatrixSection> h_sectionInfo(numSections);
        cudaMemcpy(h_sectionInfo.data(), d_sectionInfo, 
                  numSections * sizeof(MatrixSection), cudaMemcpyDeviceToHost);
        
        for (int i = 0; i < numSections; i++) {
            cudaFree(h_sectionInfo[i].d_data);
        }
        
        cudaFree(d_sections);
        cudaFree(d_sectionInfo);
    }
    
    void processSections(int* d_source, int sourceRows, int sourceCols, 
                        int* d_results = nullptr) {
        // 1. Copy sections from source matrix
        dim3 blockDim(16, 16, 1);
        dim3 gridDim(1, 1, numSections);
        
        for (int i = 0; i < numSections; i++) {
            MatrixSection info;
            cudaMemcpy(&info, &d_sectionInfo[i], sizeof(MatrixSection), cudaMemcpyDeviceToHost);
            
            gridDim.x = (info.cols + blockDim.x - 1) / blockDim.x;
            gridDim.y = (info.rows + blockDim.y - 1) / blockDim.y;
            
            copySectionsKernel<<<gridDim, blockDim>>>(
                d_source, sourceRows, sourceCols, d_sections, d_sectionInfo, numSections);
        }
        cudaDeviceSynchronize();
        
        // 2. Process sections
        for (int i = 0; i < numSections; i++) {
            MatrixSection info;
            cudaMemcpy(&info, &d_sectionInfo[i], sizeof(MatrixSection), cudaMemcpyDeviceToHost);
            
            gridDim.x = (info.cols + blockDim.x - 1) / blockDim.x;
            gridDim.y = (info.rows + blockDim.y - 1) / blockDim.y;
            
            processSectionsKernel<<<gridDim, blockDim>>>(
                d_sections, d_sectionInfo, numSections, d_results);
        }
        cudaDeviceSynchronize();
        
        // 3. Copy processed sections back to source (optional)
        for (int i = 0; i < numSections; i++) {
            MatrixSection info;
            cudaMemcpy(&info, &d_sectionInfo[i], sizeof(MatrixSection), cudaMemcpyDeviceToHost);
            
            gridDim.x = (info.cols + blockDim.x - 1) / blockDim.x;
            gridDim.y = (info.rows + blockDim.y - 1) / blockDim.y;
            
            copyBackSectionsKernel<<<gridDim, blockDim>>>(
                d_source, sourceRows, sourceCols, d_sections, d_sectionInfo, numSections);
        }
        cudaDeviceSynchronize();
    }
};

#endif // SECTION_OPS_HPP