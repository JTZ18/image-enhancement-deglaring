#!/usr/bin/env python3
import os
import sys
import argparse
from PIL import Image

def split_image(image_path, output_dir=None):
    """
    Split a horizontally stacked image into three separate images:
    1. Ground truth (left)
    2. Glared image (middle)
    3. Glare mask (right)
    """
    # Create output directory if not specified
    if output_dir is None:
        output_dir = os.path.dirname(image_path)
        if not output_dir:
            output_dir = '.'
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the image
    try:
        img = Image.open(image_path)
    except Exception as e:
        print(f"Error opening image: {e}")
        return False
    
    # Get the base filename without extension
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # Get image dimensions
    width, height = img.size
    
    # Calculate the width of each part (assuming equal widths)
    part_width = width // 3
    
    # Extract each part
    ground_truth = img.crop((0, 0, part_width, height))
    glared_image = img.crop((part_width, 0, part_width * 2, height))
    glare_mask = img.crop((part_width * 2, 0, width, height))
    
    # Save each part
    ground_truth_path = os.path.join(output_dir, f"{base_name}_ground_truth.png")
    glared_image_path = os.path.join(output_dir, f"{base_name}_glared.png")
    glare_mask_path = os.path.join(output_dir, f"{base_name}_mask.png")
    
    ground_truth.save(ground_truth_path)
    glared_image.save(glared_image_path)
    glare_mask.save(glare_mask_path)
    
    print(f"Images saved to:")
    print(f"  Ground truth: {ground_truth_path}")
    print(f"  Glared image: {glared_image_path}")
    print(f"  Glare mask: {glare_mask_path}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Split a combined image into ground truth, glared image, and glare mask.')
    parser.add_argument('image_path', help='Path to the combined image')
    parser.add_argument('--output-dir', '-o', help='Directory to save the split images (default: same as input)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image_path):
        print(f"Error: Image file not found: {args.image_path}")
        return 1
    
    success = split_image(args.image_path, args.output_dir)
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())