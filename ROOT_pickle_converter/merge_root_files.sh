#!/bin/bash

#check if the number of arguments is correct
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <input_dir>"
    exit 1
fi

# Directory containing the ROOT files
INPUT_DIR=$1
# Directory to save merged batches
OUTPUT_DIR=$INPUT_DIR
# Number of files per batch
BATCH_SIZE=20

# Create the output directory if it doesn't exist
mkdir -p "${OUTPUT_DIR}"

# Get a list of all ROOT files in the input directory
ROOT_FILES=("${INPUT_DIR}"/*.root)

# Total number of ROOT files
TOTAL_FILES=${#ROOT_FILES[@]}

# Batch counter
BATCH_NUMBER=1

# Loop through the ROOT files in batches of BATCH_SIZE
for ((i=0; i<${TOTAL_FILES}; i+=${BATCH_SIZE})); do
    # Files for the current batch
    BATCH_FILES=("${ROOT_FILES[@]:$i:$BATCH_SIZE}")

    # Output file for the current batch
    OUTPUT_FILE="$OUTPUT_DIR/batch_${BATCH_NUMBER}.root"

    # Run hadd command to merge the files
    echo "Merging batch $BATCH_NUMBER: ${BATCH_FILES[*]} -> $OUTPUT_FILE"
    hadd -f "$OUTPUT_FILE" "${BATCH_FILES[@]}"

    # Increment batch counter
    BATCH_NUMBER=$((BATCH_NUMBER + 1))
done

echo "All batches have been merged!"