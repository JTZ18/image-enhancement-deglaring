import os
import argparse
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import matplotlib.pyplot as plt
import glob

# Import from src modules
from src.model import LightweightUNet
from src.optimized_model import OptimizedUNet
from src.optimized_dataset import OptimizedGlareRemovalDataset, get_optimized_transformations
from src.utils import set_seed

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Evaluate glare removal model on validation set')
    parser.add_argument('--data_dir', type=str, default='SD1/val', help='Directory containing validation images')
    parser.add_argument('--model_path', type=str, default='./models/best_model.pth',
                      help='Path to the best model checkpoint')
    parser.add_argument('--model', type=str, help='Model architecture to use', choices=['optimized', 'lightweight'], default='lightweight')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of dataloader workers')
    parser.add_argument('--image_size', type=int, default=512, help='Size of images for evaluation')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--save_visualizations', action='store_true',
                      help='Save visualization of predictions')
    parser.add_argument('--visualizations_dir', type=str, default='./eval_visualizations',
                      help='Directory to save visualizations')
    parser.add_argument('--max_vis_samples', type=int, default=10,
                      help='Maximum number of samples to visualize')
    return parser.parse_args()

def load_model(model_path, device, model_arch):
    """
    Load the model from checkpoint

    Args:
        model_path (str): Path to the model checkpoint
        device (torch.device): Device to load the model on
        model_arch (str): Model architecture to use
    Returns:
        nn.Module: Loaded model
    """
    # Initialize model
    if model_arch == 'optimized':
        model = OptimizedUNet(in_channels=1, out_channels=1).to(device)
        print("Loaded optimized unet model")
    else:
        model = LightweightUNet(in_channels=1, out_channels=1).to(device)
        print("Loaded lightweight unet model")

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    # Handle two different checkpoint formats - full checkpoint or just state dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from epoch {checkpoint['epoch']} with validation loss {checkpoint['loss']:.4f}")
    else:
        model.load_state_dict(checkpoint)
        print("Loaded model weights (state dict only)")

    model.eval()
    return model

def create_evaluation_dataloader(data_dir, batch_size, num_workers, image_size, seed):
    """
    Create dataloader for evaluation

    Args:
        data_dir (str): Directory containing validation images
        batch_size (int): Batch size for dataloader
        num_workers (int): Number of worker threads
        image_size (int): Size to resize images to
        seed (int): Random seed for reproducibility

    Returns:
        DataLoader: DataLoader for evaluation
    """
    # Get all image file paths from validation set
    image_paths = []
    for ext in ('*.png', '*.jpg', '*.jpeg'):
        image_paths.extend(glob.glob(os.path.join(data_dir, ext)))

    if not image_paths:
        raise ValueError(f"No images found in {data_dir}")

    print(f"Found {len(image_paths)} validation images in {data_dir}")

    # Get transformations (only validation transforms needed)
    _, val_transform = get_optimized_transformations(image_size)

    # Create dataset
    val_dataset = OptimizedGlareRemovalDataset(
        image_paths,
        transform=val_transform,
        seed=seed,
        image_size=image_size,
        cache_images=False
    )

    # Create dataloader
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return val_loader

def evaluate(model, val_loader, device, save_visualizations=False,
           visualizations_dir=None, max_vis_samples=10):
    """
    Evaluate model on validation set

    Args:
        model (nn.Module): Model to evaluate
        val_loader (DataLoader): Validation data loader
        device (torch.device): Device to run evaluation on
        save_visualizations (bool): Whether to save visualization of predictions
        visualizations_dir (str): Directory to save visualizations
        max_vis_samples (int): Maximum number of samples to visualize

    Returns:
        dict: Dictionary containing evaluation metrics
    """
    # Prepare for evaluation
    model.eval()
    criterion = nn.L1Loss()
    total_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    num_batches = len(val_loader)

    # Create directory for visualizations if needed
    if save_visualizations and visualizations_dir:
        os.makedirs(visualizations_dir, exist_ok=True)

    # Counter for visualization samples
    vis_count = 0

    print("Starting evaluation...")
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(tqdm(val_loader, desc="Evaluating")):
            # Move to device
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)
            outputs_min = outputs.min().item()
            outputs_max = outputs.max().item()
            # print(f"[Batch {batch_idx}] Output tensor range: min={outputs_min:.4f}, max={outputs_max:.4f}")

            # Calculate L1 loss
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            # Calculate additional metrics (on CPU for skimage functions)
            outputs_np = outputs.detach().cpu().numpy()
            targets_np = targets.detach().cpu().numpy()

            # Calculate metrics for each image in batch
            for i in range(inputs.size(0)):
                output_img = outputs_np[i, 0]  # [H, W]
                target_img = targets_np[i, 0]  # [H, W]

                # Ensure correct range for metrics (0 to 1)
                output_img = np.clip(output_img, 0, 1)

                # Calculate PSNR
                psnr = peak_signal_noise_ratio(target_img, output_img, data_range=1.0)
                total_psnr += psnr

                # Calculate SSIM
                ssim = structural_similarity(target_img, output_img, data_range=1.0)
                total_ssim += ssim

                # Save visualizations for a few samples
                if save_visualizations and vis_count < max_vis_samples:
                    input_img = inputs.cpu().numpy()[i, 0]  # [H, W]

                    plt.figure(figsize=(15, 5))

                    plt.subplot(1, 3, 1)
                    plt.imshow(input_img, cmap='gray')
                    input_min = input_img.min()
                    input_max = input_img.max()
                    plt.title(f'Input\nRange: [{input_min:.2f}, {input_max:.2f}]')
                    plt.axis('off')

                    plt.subplot(1, 3, 2)
                    plt.imshow(output_img, cmap='gray')
                    output_min = output_img.min()
                    output_max = output_img.max()
                    plt.title(f'Prediction\nPSNR: {psnr:.2f}, SSIM: {ssim:.4f}\nRange: [{output_min:.2f}, {output_max:.2f}]')
                    plt.axis('off')

                    plt.subplot(1, 3, 3)
                    plt.imshow(target_img, cmap='gray')
                    target_min = target_img.min()
                    target_max = target_img.max()
                    plt.title(f'Ground Truth\nRange: [{target_min:.2f}, {target_max:.2f}]')
                    plt.axis('off')

                    plt.tight_layout()
                    plt.savefig(os.path.join(visualizations_dir, f'sample_{vis_count}.png'))
                    plt.close()

                    vis_count += 1

    # Calculate average metrics
    total_samples = len(val_loader.dataset)
    avg_loss = total_loss / num_batches
    avg_psnr = total_psnr / total_samples
    avg_ssim = total_ssim / total_samples

    # Print metrics
    print(f"Evaluation on {total_samples} samples:")
    print(f"L1 Loss: {avg_loss:.4f}")
    print(f"PSNR: {avg_psnr:.2f} dB")
    print(f"SSIM: {avg_ssim:.4f}")

    # Return metrics as dictionary
    return {
        'l1_loss': avg_loss,
        'psnr': avg_psnr,
        'ssim': avg_ssim
    }

def main():
    """Main function for evaluating the model"""
    args = parse_args()

    # Set random seed for reproducibility
    set_seed(args.seed)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    model = load_model(args.model_path, device, args.model)

    # Create evaluation dataloader
    val_loader = create_evaluation_dataloader(
        args.data_dir,
        args.batch_size,
        args.num_workers,
        args.image_size,
        args.seed
    )

    # Evaluate model
    metrics = evaluate(
        model,
        val_loader,
        device,
        save_visualizations=args.save_visualizations,
        visualizations_dir=args.visualizations_dir,
        max_vis_samples=args.max_vis_samples
    )

    # Save metrics to file
    metrics_dir = os.path.dirname(args.model_path)
    with open(os.path.join(metrics_dir, 'evaluation_results.txt'), 'w') as f:
        f.write(f"Evaluation results on {args.data_dir}:\n")
        f.write(f"L1 Loss: {metrics['l1_loss']:.4f}\n")
        f.write(f"PSNR: {metrics['psnr']:.2f} dB\n")
        f.write(f"SSIM: {metrics['ssim']:.4f}\n")

    print(f"Evaluation completed. Results saved to {os.path.join(metrics_dir, 'evaluation_results.txt')}")

if __name__ == '__main__':
    main()