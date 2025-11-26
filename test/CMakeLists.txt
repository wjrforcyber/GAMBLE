set(TEST_CASES_DIR "${CMAKE_SOURCE_DIR}/cases")

add_executable(thrustSum thrustSum.cu)
add_test(NAME thrustSumMatrix COMMAND thrustSum)

add_executable(normalize normalize.cu)
add_test(NAME normalizeMatrix COMMAND normalize)

#add_executable(sectionOutGPU sectionOutGPU.cu ../sectionProcessGPU.cu)
#add_test(NAME sectionOutMatrixGPU COMMAND sectionOutGPU)

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

add_executable(checkViaMap checkViaMap.cpp ../source/util.cpp)
add_test(NAME checkViaPathMap COMMAND checkViaMap)

add_executable(checkSpecialSource checkSpecialSource.cpp ../source/util.cpp)
add_test(NAME checkSpecialMatrixSource COMMAND checkSpecialSource)

add_executable(iterAllPins iterAllPins.cpp ../source/util.cpp)
add_test(NAME iterAllMatrixPins COMMAND iterAllPins)

add_executable(test10Condition test10Condition.cpp ../source/util.cpp)
add_test(NAME testMatrix10Condition COMMAND test10Condition)

add_executable(loadTestCasesCPU loadTestCases.cpp ../source/util.cpp)
add_test(NAME loadMatrixTestCasesCPU COMMAND loadTestCasesCPU)

set_tests_properties(loadMatrixTestCasesCPU PROPERTIES
    ENVIRONMENT "TEST_CASES_DIR=${TEST_CASES_DIR}"
)

add_executable(sectionOutRoute sectionOutRoute.cu ../sectionProcessGPU.cu ../source/util.cpp)
add_test(NAME sectionMatrixOutRoute COMMAND sectionOutRoute)