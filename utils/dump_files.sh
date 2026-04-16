#!/bin/bash

# Script to dump multiple file contents into /tmp/dump_files
# Each file content is prefixed with "# <file_path>" followed by empty line

OUTPUT_FILE="/tmp/dump_files"

# Clear/create the output file
> "$OUTPUT_FILE"

# Process each file path provided as arguments
for file_path in "$@"; do
    # Add the file header
    echo "# $file_path" >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"
    
    # Add the file contents
    cat "$file_path" >> "$OUTPUT_FILE"
    
    # Add empty line between files
    echo "" >> "$OUTPUT_FILE"
done

echo "Dumped ${#@} files to $OUTPUT_FILE"
