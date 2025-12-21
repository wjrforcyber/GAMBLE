#!/bin/bash

# Read the input file (pipe input or file argument)
input_file="${1:-/dev/stdin}"

# Print the table header
echo "| Cases | CPU_time | CPU_cost | GPU_time | GPU_cost |"
echo "| --- | --- | --- | --- | --- |"

# Process each CASE block
awk '
BEGIN {
    # Initialize variables
    case_name = ""
    cpu_time = ""
    cpu_cost = ""
    gpu_time = ""
    gpu_cost = ""
}

/^=== CASE [0-9]+ ===$/ {
    # If we have collected data from previous case, print it
    if (case_name != "") {
        printf "| `%s` | %s | %s | %s | %s |\n", 
               case_name, cpu_time, cpu_cost, gpu_time, gpu_cost
    }
    
    # Reset for new case
    case_name = ""
    cpu_time = ""
    cpu_cost = ""
    gpu_time = ""
    gpu_cost = ""
}

/Test file:.*\.txt$/ {
    # Extract case filename
    split($0, parts, "/")
    case_name = parts[length(parts)]
}

/Original CPU version:/ {
    getline
    if ($0 ~ /time:/) {
        # Extract CPU time (remove leading spaces and "time:")
        cpu_time = $2
        # Remove trailing comma if present
        gsub(/,$/, "", cpu_time)
    }
    getline
    if ($0 ~ /cost:/) {
        # Extract CPU cost
        cpu_cost = $2
    }
}

/Current GPU version:/ {
    getline
    if ($0 ~ /time:/) {
        # Extract GPU time (remove leading spaces and "time:")
        gpu_time = $2
        # Remove trailing comma if present
        gsub(/,$/, "", gpu_time)
    }
    getline
    if ($0 ~ /cost:/) {
        # Extract GPU cost
        gpu_cost = $2
    }
}

END {
    # Print the last case
    if (case_name != "") {
        printf "| `%s` | %s | %s | %s | %s |\n", 
               case_name, cpu_time, cpu_cost, gpu_time, gpu_cost
    }
}
' "$input_file"