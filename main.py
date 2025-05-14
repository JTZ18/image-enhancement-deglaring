import os
import argparse
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Import modules from src
from src.model import LightweightUNet, get_model_size_mb
from src.preprocess import preprocess_inference, postprocess_output


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='De-glare images using the trained model')
    parser.add_argument('--input', type=str, help='Path to input image or directory')
    parser.add_argument('--output_dir', type=str, default='./results', help='Directory to save results')
    parser.add_argument('--model_path', type=str, default='./models/model_weights.pth', help='Path to model weights')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for processing')
    parser.add_argument('--visualize', action='store_true', help='Visualize results')
    return parser.parse_args()


def process_single_image(model, image_path, device):
    """Process a single image for de-glaring"""
    # Preprocess the image
    input_tensor = preprocess_inference(image_path)
    input_tensor = input_tensor.to(device)
    
    # Model inference
    with torch.no_grad():
        output_tensor = model(input_tensor)
    
    # Postprocess the output
    output_image = postprocess_output(output_tensor)
    
    return output_image


def visualize_results(input_path, output_image, output_path):
    """Visualize and save input and output images side by side"""
    # Load the original image
    input_image = np.array(Image.open(input_path).convert('L'))
    
    # Create figure with two subplots
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    
    # Display input image
    ax[0].imshow(input_image, cmap='gray')
    ax[0].set_title('Input Image (with glare)')
    ax[0].axis('off')
    
    # Display output image
    ax[1].imshow(output_image, cmap='gray')
    ax[1].set_title('De-glared Image')
    ax[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path.replace('.png', '_comparison.png'))
    plt.close()


def main():
    """Main function for running the de-glaring model"""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = LightweightUNet(in_channels=1, out_channels=1)
    
    # Check if model file exists
    if not os.path.exists(args.model_path):
        print(f"Model file not found: {args.model_path}")
        print("You need to train the model first using train.py")
        return
    
    # Load model weights
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Print model info
    model_size_mb = get_model_size_mb(model)
    print(f"Model loaded successfully - Size: {model_size_mb:.2f} MB")
    
    # Check if input is a file or directory
    if os.path.isfile(args.input):
        # Process single image
        print(f"Processing image: {args.input}")
        output_image = process_single_image(model, args.input, device)
        
        # Save output image
        output_path = os.path.join(args.output_dir, os.path.basename(args.input))
        Image.fromarray(output_image).save(output_path)
        print(f"Output saved to: {output_path}")
        
        # Visualize results if requested
        if args.visualize:
            visualize_results(args.input, output_image, output_path)
            print(f"Visualization saved to: {output_path.replace('.png', '_comparison.png')}")
    
    elif os.path.isdir(args.input):
        # Process all images in directory
        image_files = [
            os.path.join(args.input, f) for f in os.listdir(args.input) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        
        print(f"Found {len(image_files)} images to process")
        
        for image_path in image_files:
            # Process image
            print(f"Processing image: {image_path}")
            output_image = process_single_image(model, image_path, device)
            
            # Save output image
            output_path = os.path.join(args.output_dir, os.path.basename(image_path))
            Image.fromarray(output_image).save(output_path)
            
            # Visualize results if requested
            if args.visualize:
                visualize_results(image_path, output_image, output_path)
        
        print(f"All images processed and saved to: {args.output_dir}")
    
    else:
        print(f"Input path not found: {args.input}")


if __name__ == "__main__":
    main()