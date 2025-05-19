#!/usr/bin/env python3
import os
import sys
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import numpy as np

def check_png_dimensions(data_dir="SD1", required_width=1536, required_height=512):
    """
    Check if all PNG files in the train and val folders of data_dir have the required dimensions,
    RGBA format (4 channels), and valid grayscale pixel values between 0-255.
    
    Args:
        data_dir (str): Path to the main directory (contains train and val subdirectories)
        required_width (int): Required width for PNG files
        required_height (int): Required height for PNG files
    
    Returns:
        tuple: (incorrect_dimensions, invalid_channels, invalid_pixels, total_files)
            - incorrect_dimensions: List of tuples (relative_path, actual_width, actual_height) for files with incorrect dimensions
            - invalid_channels: List of tuples (relative_path, actual_channels) for files without 4 channels (RGBA)
            - invalid_pixels: List of relative_paths for files with grayscale pixel values outside [0, 255]
            - total_files: Total number of PNG files checked
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
    
    incorrect_dimensions = []
    invalid_channels = []
    invalid_pixels = []
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
        
        # Check dimensions of each PNG file with progress bar
        print(f"Checking {len(png_files)} PNG files in '{subdir_path}'...")
        for png_file in tqdm(png_files, desc=f"Validating {subdir}", unit="file"):
            try:
                with Image.open(png_file) as img:
                    # Check dimensions
                    width, height = img.size
                    rel_path = f"{subdir}/{png_file.name}"
                    
                    if width != required_width or height != required_height:
                        # Store relative path for cleaner output
                        incorrect_dimensions.append((rel_path, width, height))
                    
                    # Check for RGBA format (4 channels)
                    if img.mode != 'RGBA':
                        invalid_channels.append((rel_path, img.mode))
                    
                    # Check pixel values after grayscale conversion
                    grayscale = img.convert('L')  # Convert to grayscale
                    pixels = list(grayscale.getdata())
                    
                    # Check if all pixel values are within [0, 255]
                    if any(p < 0 or p > 255 for p in pixels):
                        invalid_pixels.append(rel_path)
            except Exception as e:
                print(f"Error processing {png_file}: {e}")
    
    return incorrect_dimensions, invalid_channels, invalid_pixels, total_files

def main():
    print("\nStarting PNG validation checks...")
    print("\nChecking for: ")
    print("✅ Dimensions: 1536x512")
    print("✅ Format: RGBA (4 channels)")
    print("✅ Grayscale pixel values: between [0, 255]")
    print("\n" + "-"*50)
    
    incorrect_dimensions, invalid_channels, invalid_pixels, total_files = check_png_dimensions()
    
    print("\n" + "-"*50)
    print(f"Checked {total_files} PNG files in SD1/train and SD1/val folders")
    
    # Report dimension errors
    if not incorrect_dimensions:
        print("✓ All PNG files have the correct dimensions (1536x512)")
    else:
        print(f"✗ Found {len(incorrect_dimensions)} files with incorrect dimensions:")
        for rel_path, width, height in incorrect_dimensions:
            print(f"  SD1/{rel_path}: {width}x{height} (should be 1536x512)")
    
    # Report channel errors
    if not invalid_channels:
        print("✓ All PNG files have the correct format (RGBA with 4 channels)")
    else:
        print(f"✗ Found {len(invalid_channels)} files with incorrect format:")
        for rel_path, mode in invalid_channels:
            print(f"  SD1/{rel_path}: {mode} (should be RGBA)")
    
    # Report pixel value errors
    if not invalid_pixels:
        print("✓ All PNG files have valid grayscale pixel values [0-255]")
    else:
        print(f"✗ Found {len(invalid_pixels)} files with invalid grayscale pixel values:")
        for rel_path in invalid_pixels:
            print(f"  SD1/{rel_path}: contains values outside range [0-255]")
    
    # Overall summary
    if not (incorrect_dimensions or invalid_channels or invalid_pixels):
        print("\nAll checks passed! The dataset is ready for processing.")
    else:
        print("\nSome checks failed. Please fix the issues before proceeding.")

if __name__ == "__main__":
    main()