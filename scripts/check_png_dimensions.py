#!/usr/bin/env python3
import os
import sys
from PIL import Image
from pathlib import Path

def check_png_dimensions(data_dir="SD1", required_width=1536, required_height=512):
    """
    Check if all PNG files in the train and val folders of data_dir have the required dimensions.
    
    Args:
        data_dir (str): Path to the main directory (contains train and val subdirectories)
        required_width (int): Required width for PNG files
        required_height (int): Required height for PNG files
    
    Returns:
        list: List of tuples (relative_path, actual_width, actual_height) for files with incorrect dimensions
    """
    data_path = Path(data_dir)
    
    # Check if directory exists
    if not data_path.exists():
        print(f"Error: Directory '{data_dir}' does not exist")
        sys.exit(1)
    
    if not data_path.is_dir():
        print(f"Error: '{data_dir}' is not a directory")
        sys.exit(1)
    
    # Expected subdirectories
    subdirs = ["train", "val"]
    
    incorrect_files = []
    total_files = 0
    
    for subdir in subdirs:
        subdir_path = data_path / subdir
        
        if not subdir_path.exists():
            print(f"Warning: Subdirectory '{subdir}' does not exist in '{data_dir}'")
            continue
            
        # Find all PNG files in the subdirectory
        png_files = list(subdir_path.glob("*.png"))
        total_files += len(png_files)
        
        if not png_files:
            print(f"No PNG files found in '{subdir_path}'")
            continue
        
        # Check dimensions of each PNG file
        for png_file in png_files:
            try:
                with Image.open(png_file) as img:
                    width, height = img.size
                    if width != required_width or height != required_height:
                        # Store relative path for cleaner output
                        rel_path = f"{subdir}/{png_file.name}"
                        incorrect_files.append((rel_path, width, height))
            except Exception as e:
                print(f"Error processing {png_file}: {e}")
    
    return incorrect_files, total_files

def main():
    incorrect_files, total_files = check_png_dimensions()
    
    print(f"Checked {total_files} PNG files in SD1/train and SD1/val folders")
    
    if not incorrect_files:
        print("All PNG files have the correct dimensions (1536x512)")
    else:
        print(f"Found {len(incorrect_files)} files with incorrect dimensions:")
        for rel_path, width, height in incorrect_files:
            print(f"  SD1/{rel_path}: {width}x{height} (should be 1536x512)")

if __name__ == "__main__":
    main()