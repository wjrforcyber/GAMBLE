#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#ifndef SECTION_OPS_HPP
#define SECTION_OPS_HPP

struct MatrixSection {
    int top, left;      // Upper-left corner
    int bottom, right;  // Lower-right corner
    int* d_data;        // Device pointer for this section
    int rows, cols;     // Section dimensions
};

__global__ void copySectionsKernel(int* source, int sourceRows, int sourceCols,
    int** sections, MatrixSection* sectionInfo, int numSections);

__global__ void processSectionsKernel(int** sections, MatrixSection* sectionInfo, 
    int numSections, int* results);


class MatrixSectionProcessor {
private:
    int** d_sections;
    MatrixSection* d_sectionInfo;
    int numSections;
    std::vector<MatrixSection> h_sectionInfo; // Store host copy
    
public:
    MatrixSectionProcessor(const std::vector<MatrixSection>& sections) {
        numSections = sections.size();
        h_sectionInfo = sections; // Store original sections
        
        // Calculate dimensions for each section
        for (int i = 0; i < numSections; i++) {
            h_sectionInfo[i].rows = sections[i].bottom - sections[i].top + 1;
            h_sectionInfo[i].cols = sections[i].right - sections[i].left + 1;
        }
        
        // Allocate device memory for section pointers and info
        cudaMalloc(&d_sections, numSections * sizeof(int*));
        cudaMalloc(&d_sectionInfo, numSections * sizeof(MatrixSection));
        
        // Allocate device memory for each section and update host info
        std::vector<int*> h_sectionPtrs(numSections);
        for (int i = 0; i < numSections; i++) {
            size_t sectionSize = h_sectionInfo[i].rows * h_sectionInfo[i].cols * sizeof(int);
            cudaMalloc(&h_sectionInfo[i].d_data, sectionSize);
            h_sectionPtrs[i] = h_sectionInfo[i].d_data;
        }
        
        // Copy section info to device (without the d_data pointers)
        std::vector<MatrixSection> sectionInfoNoPointers = h_sectionInfo;
        cudaMemcpy(d_sectionInfo, sectionInfoNoPointers.data(), 
                  numSections * sizeof(MatrixSection), cudaMemcpyHostToDevice);
        
        // Copy section pointers array
        cudaMemcpy(d_sections, h_sectionPtrs.data(), 
                  numSections * sizeof(int*), cudaMemcpyHostToDevice);
    }
    
    ~MatrixSectionProcessor() {
        // Free section device memory using host info
        for (int i = 0; i < numSections; i++) {
            cudaFree(h_sectionInfo[i].d_data);
        }
        cudaFree(d_sections);
        cudaFree(d_sectionInfo);
    }
    
    void processSections(int* d_source, int sourceRows, int sourceCols, 
                        int* d_results = nullptr) {
        // 1. Copy sections from source matrix
        copySectionsKernel<<<1, 1>>>(d_source, sourceRows, sourceCols, 
                                    d_sections, d_sectionInfo, numSections);
        cudaDeviceSynchronize();
        
        // 2. Process sections  
        processSectionsKernel<<<1, 1>>>(d_sections, d_sectionInfo, numSections, d_results);
        cudaDeviceSynchronize();
    }
    
    void printSections() {
        std::vector<int*> h_sectionData(numSections);
        
        // Copy section data from device to host
        for (int i = 0; i < numSections; i++) {
            size_t sectionSize = h_sectionInfo[i].rows * h_sectionInfo[i].cols * sizeof(int);
            h_sectionData[i] = new int[h_sectionInfo[i].rows * h_sectionInfo[i].cols];
            
            cudaMemcpy(h_sectionData[i], h_sectionInfo[i].d_data, sectionSize, 
                      cudaMemcpyDeviceToHost);
            
            std::cout << "\nSection " << i + 1 << " (" 
                      << h_sectionInfo[i].top << "," << h_sectionInfo[i].left << ") to ("
                      << h_sectionInfo[i].bottom << "," << h_sectionInfo[i].right << "): "
                      << h_sectionInfo[i].rows << "x" << h_sectionInfo[i].cols << std::endl;
            
            for (int row = 0; row < h_sectionInfo[i].rows; row++) {
                for (int col = 0; col < h_sectionInfo[i].cols; col++) {
                    std::cout << h_sectionData[i][row * h_sectionInfo[i].cols + col] << " ";
                }
                std::cout << std::endl;
            }
        }
        
        // Clean up
        for (int i = 0; i < numSections; i++) {
            delete[] h_sectionData[i];
        }
    }
};

#endif // SECTION_OPS_HPP