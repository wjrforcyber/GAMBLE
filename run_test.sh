#!/bin/bash

LOG_FILE="test_results.log"
SUMMARY_FILE="test_summary.txt"

# Clear previous logs
> "$LOG_FILE"
> "$SUMMARY_FILE"

# Header
echo "TEST EXECUTION REPORT" | tee -a "$SUMMARY_FILE"
echo "Start time: $(date)" | tee -a "$SUMMARY_FILE"
echo "========================================" | tee -a "$SUMMARY_FILE"

PASS_COUNT=0
FAIL_COUNT=0
TOTAL_TIME=0

for i in {1..13}; do 
    if [ -f "../cases/case${i}.txt" ]; then 
        echo "=== CASE $i ===" >> "$LOG_FILE"
        echo "Test file: ../cases/case${i}.txt" >> "$LOG_FILE"
        
        # Run test and capture output and time
        START_TIME=$(date +%s.%N)
        ./loadTestCases < "../cases/case${i}.txt" >> "$LOG_FILE" 2>&1
        EXIT_CODE=$?
        END_TIME=$(date +%s.%N)
        
        # Calculate elapsed time
        ELAPSED=$(echo "$END_TIME - $START_TIME" | bc)
        TOTAL_TIME=$(echo "$TOTAL_TIME + $ELAPSED" | bc)
        
        # Log results
        echo "Exit code: $EXIT_CODE" >> "$LOG_FILE"
        echo "Execution time: ${ELAPSED}s" >> "$LOG_FILE"
        echo "----------------------------------------" >> "$LOG_FILE"
        
        # Update summary
        if [ $EXIT_CODE -eq 0 ]; then
            STATUS="PASS ✓"
            ((PASS_COUNT++))
        else
            STATUS="FAIL ✗ (code: $EXIT_CODE)"
            ((FAIL_COUNT++))
        fi
        
        printf "Case %2d: %-20s Time: %.2fs\n" $i "$STATUS" $ELAPSED | tee -a "$SUMMARY_FILE"
    else
        echo "Case $i: SKIPPED (file not found)" | tee -a "$SUMMARY_FILE"
    fi
done

# Summary
echo "========================================" | tee -a "$SUMMARY_FILE"
echo "SUMMARY:" | tee -a "$SUMMARY_FILE"
echo "Total cases: 13" | tee -a "$SUMMARY_FILE"
echo "Executed: $((PASS_COUNT + FAIL_COUNT))" | tee -a "$SUMMARY_FILE"
echo "Passed: $PASS_COUNT" | tee -a "$SUMMARY_FILE"
echo "Failed: $FAIL_COUNT" | tee -a "$SUMMARY_FILE"
echo "Skipped: $((13 - PASS_COUNT - FAIL_COUNT))" | tee -a "$SUMMARY_FILE"
printf "Total execution time: %.2fs\n" $TOTAL_TIME | tee -a "$SUMMARY_FILE"
echo "End time: $(date)" | tee -a "$SUMMARY_FILE"
echo "========================================" | tee -a "$SUMMARY_FILE"
echo "Detailed log: $LOG_FILE" | tee -a "$SUMMARY_FILE"