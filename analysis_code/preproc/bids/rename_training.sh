#!/bin/bash

# Check if directory argument is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <directory>"
    exit 1
fi

# Set the directory containing the files
directory="$1"

# Set the replacement string
replacement="Training"

# Navigate to the directory
cd "$directory" || exit

# Loop through all files in the directory
for file in *; do
    # Check if the file name contains "_training"
    if [[ "$file" == *"_training"* ]]; then
        # Perform the renaming
        new_name=$(echo "$file" | sed "s/_training/$replacement/")
        mv "$file" "$new_name"
        echo "Renamed $file to $new_name"
    fi
done
