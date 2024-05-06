#!/bin/bash

# Source and target directories
source_dir="./022.OCR_데이터(옛한글)/01-1.정식개방데이터/Training/01.원천데이터"
target_dir="./train/images2"

# Create target directory if it doesn't exist
mkdir -p "$target_dir"

# Loop through all zip files in the source directory
for file in "$source_dir"/*.zip; do
    # Check if the file exists to avoid errors in case there are no zip files
    if [ -f "$file" ]; then
        echo "Unzipping $file into $target_dir"
        unzip -O cp949 -d "$target_dir" "$file"
    else
        echo "No zip files found in $source_dir."
        break
    fi
done
