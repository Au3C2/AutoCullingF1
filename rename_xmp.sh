#!/bin/bash

# rename_xmp.sh - Rename *.JPG.xmp back to *.xmp
# Usage: ./rename_xmp.sh /path/to/directory

TARGET_DIR="$1"

if [ -z "$TARGET_DIR" ]; then
    echo "Usage: $0 /path/to/directory"
    exit 1
fi

if [ ! -d "$TARGET_DIR" ]; then
    echo "Error: Directory $TARGET_DIR does not exist."
    exit 1
fi

echo "Renaming *.JPG.xmp to *.xmp in $TARGET_DIR ..."

count=0
# Use a loop to handle filenames with spaces
for f in "$TARGET_DIR"/*.JPG.xmp; do
    # Check if any files matched the pattern
    [ -e "$f" ] || continue
    
    # New name: replace .JPG.xmp with .xmp
    new_name="${f/.JPG.xmp/.xmp}"
    
    mv "$f" "$new_name"
    ((count++))
done

echo "Done. Renamed $count files."
