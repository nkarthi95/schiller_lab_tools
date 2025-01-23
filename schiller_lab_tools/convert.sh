#!/bin/bash

# Set the directory to iterate through (defaults to the current directory if not specified)
DIRECTORY=${1:-.}

# Iterate through all folders in the specified directory
for folder in "$DIRECTORY"/*; do
    if [ -d "$folder" ]; then
        echo "Folder: $(basename "$folder")"
        cd $folder
        jupyter nbconvert --to script *.ipynb
        cd ..
    fi
done