#!/bin/bash

# Define the base command
BASE_CMD="./finetune.py"

# Define the parameter combinations
PARAMS=(
    ""
    "--log1p"
    "--normalize_total"
    "--normalize_total --log1p"
    "--normalize_total --normalize_CPM --log1p"
    "--normalize_total --normalize_CPM --exclude_highly_expressed --log1p"
)

# Loop through each parameter combination and execute the script
for PARAM in "${PARAMS[@]}"; do
    echo "Executing: CUDA_VISIBLE_DEVICES=0 $BASE_CMD $PARAM"
    CUDA_VISIBLE_DEVICES=0 $BASE_CMD $PARAM
    if [ $? -ne 0 ]; then
        echo "Error: Command failed for parameters: $PARAM"
        exit 1
    fi
    echo "------------------------------------------"
done

echo "All commands executed successfully!"