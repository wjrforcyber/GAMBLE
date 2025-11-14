add_executable(thrustSum thrustSum.cu)
add_test(NAME thrustSumMatrix COMMAND thrustSum)

add_executable(normalize normalize.cu)
add_test(NAME normalizeMatrix COMMAND normalize)

add_executable(sectionOutGPU sectionOutGPU.cu ../sectionProcessGPU.cu)
add_test(NAME sectionOutMatrixGPU COMMAND sectionOutGPU)

add_executable(sectionOutCPU sectionOutCPU.cpp)
add_test(NAME sectionOutMatrixCPU COMMAND sectionOutCPU)

add_executable(utilSecDiag utilSecDiag.cpp ../source/util.cpp)
add_test(NAME utilMatrixSecDiagCPU COMMAND utilSecDiag)

add_executable(getPartitionOnSecDiag getPartitionOnSecDiag.cpp ../source/util.cpp)
add_test(NAME getPartitionOnMatrixSecDiag COMMAND getPartitionOnSecDiag)

add_executable(Sep0PartitionOnSecDiag getPartitionOnSecDiagSpec0.cpp ../source/util.cpp)
add_test(NAME Sep0PartitionOnMatrixSecDiag COMMAND Sep0PartitionOnSecDiag)

add_executable(perSectionBest perSectionBest.cpp ../source/util.cpp)
add_test(NAME perSectionMatrixBest COMMAND perSectionBest)

add_executable(specialLocation specialLocation.cpp ../source/util.cpp)
add_test(NAME specialMatrixLocation COMMAND specialLocation)

add_executable(reversedCase reversedCase.cpp ../source/util.cpp)
add_test(NAME reversedMatrixCase COMMAND reversedCase)

add_executable(fitUnconnected fitUnconnected.cpp ../source/util.cpp)
add_test(NAME fitMatrixUnconnected COMMAND fitUnconnected)

add_executable(sameRowCol sameRowCol.cpp ../source/util.cpp)
add_test(NAME sameMatrixRowCol COMMAND sameRowCol)

add_executable(matrixFixedPartition matrixFixedPartition.cpp ../source/util.cpp)
add_test(NAME matrixFixedMatrixPartition COMMAND matrixFixedPartition)