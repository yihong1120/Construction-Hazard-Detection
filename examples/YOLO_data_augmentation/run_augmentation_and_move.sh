#!/bin/bash

# Set the working directory to the directory of this script
cd "$(dirname "$0")"

# Execute the data augmentation script
echo "Starting data augmentation..."
python ../src/data_augmentation.py
echo "Data augmentation completed."

# Determine the last file number in the target directories
last_image=$(ls -v ../dataset/train/images/ | tail -n 1)
last_label=$(ls -v ../dataset/train/labels/ | tail -n 1)

# Extract the number from the file name using regular expressions
[[ $last_image =~ ([0-9]+) ]]
last_num_image=${BASH_REMATCH[1]}

[[ $last_label =~ ([0-9]+) ]]
last_num_label=${BASH_REMATCH[1]}

# Calculate the starting number for the next file
next_num=$((last_num_image > last_num_label ? last_num_image : last_num_label))
next_num=$((next_num + 1))

# Move and rename augmented images and label files
echo "Starting to move and rename augmented images and label files..."
for file in dataset_aug/train/images/*; do
    filename=$(basename -- "$file")
    extension="${filename##*.}"
    mv "$file" "../dataset/train/images/$(printf "%06d.%s" "$next_num" "$extension")"
    mv "../dataset_aug/train/labels/${filename%.*}.txt" "../dataset/train/labels/$(printf "%06d.txt" "$next_num")"
    next_num=$((next_num + 1))
done
echo "File movement and renaming completed."

# Optional: Clean the dataset_aug directory
echo "Cleaning augmentation data directory..."
rm -rf ../dataset_aug/train/images/*
rm -rf ../dataset_aug/train/labels/*
echo "Augmentation data directory cleaned."
