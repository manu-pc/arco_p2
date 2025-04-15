#!/bin/bash

# Output CSV file
OUTPUT_FILE="resultados.csv"

# Create the CSV file with a header
echo "Version,Run,Input,Output" > "$OUTPUT_FILE"

# List of source files
versions=("v1" "v2" "v3")

# Input values to test
inputs=(250 2500 5000)

# Compile each C file
for version in "${versions[@]}"; do
    make "$version" || { echo "Compilation failed for $version.c"; exit 1; }
done

# Run each executable 15 times for each input
for version in "${versions[@]}"; do
    for input in "${inputs[@]}"; do
        for i in {1..15}; do
            output=$("./$version" "$input")  # Capture output
            line="$version,$i,$input,$output"
            echo "$line" | tee -a "$OUTPUT_FILE"  # Print to console and append to CSV
        done
    done
done

echo "Execution completed. Results saved in $OUTPUT_FILE."

