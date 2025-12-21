#!/bin/bash

# Read the input file (pipe input or file argument)
input_file="${1:-/dev/stdin}"

# Start the LaTeX table
echo '\begin{table}[]'
echo '    \begin{tabular}{|c|c|c|c|c|}'
echo '    \hline'
echo '    Cases  & CPU Time  & CPU Cost & GPU Time  & GPU Cost \\\\ \hline'

# Process each CASE block
awk '
BEGIN {
    # Initialize variables
    case_num = ""
    case_name = ""
    cpu_time = ""
    cpu_cost = ""
    gpu_time = ""
    gpu_cost = ""
}

/^=== CASE ([0-9]+) ===$/ {
    # Extract case number from the line
    if (match($0, /CASE ([0-9]+)/, arr)) {
        case_num = arr[1]
    }
}

/Test file:.*\.txt$/ {
    # Extract case filename and format as CaseX
    split($0, parts, "/")
    filename = parts[length(parts)]
    # Remove .txt extension and capitalize Case
    sub(/\.txt$/, "", filename)
    case_name = "Case" case_num
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

/^----------------------------------------$/ {
    # At the end of each case block, print the data
    if (case_name != "") {
        printf "    %-6s & %-9s & %-8s & %-9s & %-8s \\\\ \\hline\n", 
               case_name, cpu_time, cpu_cost, gpu_time, gpu_cost
    }
    
    # Reset for next case
    case_name = ""
    cpu_time = ""
    cpu_cost = ""
    gpu_time = ""
    gpu_cost = ""
}

END {
    # Print the last case if not already printed
    if (case_name != "") {
        printf "    %-6s & %-9s & %-8s & %-9s & %-8s \\\\ \\hline\n", 
               case_name, cpu_time, cpu_cost, gpu_time, gpu_cost
    }
}
' "$input_file"

# End the LaTeX table
echo '    \end{tabular}'
echo '    \caption{Time consumption comparison.}'
echo '    \label{Cures}'
echo '\end{table}'