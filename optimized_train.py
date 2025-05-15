import os
import argparse
import copy
import torch
import torch.nn as nn
from torch.utils.data import random_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import wandb
from torch.amp import GradScaler, autocast
import math
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from functools import partial
import random

# Import dotenv to load environment variables
from dotenv import load_dotenv
load_dotenv()

# Import from src modules
from src.dataset import create_dataloaders, create_optimized_dataloaders
from src.model import LightweightUNet, EnhancedUNet, OptimizedUNet, count_parameters, get_model_size_mb
from src.utils import set_seed


def seed_worker(worker_id):
    """Set random seed for dataloader workers to ensure reproducibility"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train glare removal model')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing the dataset')
    parser.add_argument('--output_dir', type=str, default='./models', help='Directory to save model')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--val_split', type=float, default=0.2, help='Validation split ratio')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of dataloader workers')
    parser.add_argument('--save_every', type=int, default=10, help='Save checkpoint every N epochs')
    parser.add_argument('--early_stop', type=float, default=0, help='Early stopping loss threshold')
    parser.add_argument('--model', type=str, default='optimized', choices=['basic', 'enhanced', 'optimized'], help='Model architecture to use')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases for logging')
    parser.add_argument('--wandb_project', type=str, default='image-deglaring', help='Weights & Biases project name')
    parser.add_argument('--wandb_entity', type=str, default=None, help='Weights & Biases entity (team) name')
    parser.add_argument('--use_amp', action='store_true', help='Use automatic mixed precision training')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay for optimizer')
    parser.add_argument('--clip_grad_norm', type=float, default=1.0, help='Gradient clipping norm value')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--log_images_every', type=int, default=5, help='Log images to wandb every N epochs')
    parser.add_argument('--image_size', type=int, default=256, help='Size of images for training (smaller = faster)')
    parser.add_argument('--validation_metrics_every', type=int, default=5, help='Calculate validation metrics every N epochs')
    parser.add_argument('--prefetch_factor', type=int, default=2, help='Number of batches to prefetch per worker')
    parser.add_argument('--persistent_workers', action='store_true', help='Keep workers alive between epochs')
    return parser.parse_args()


def save_model(model, optimizer, epoch, loss, output_dir, filename):
    """Save model checkpoint"""
    os.makedirs(output_dir, exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, os.path.join(output_dir, filename))
    print(f"Checkpoint saved to {os.path.join(output_dir, filename)}")


def plot_losses(train_losses, val_losses, output_dir):
    """Plot and save training and validation loss curves"""
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('L1 Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.grid(True)

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'loss_plot.png'))
    plt.close()


def calculate_metrics(outputs, targets):
    """
    Calculate image quality metrics for the validation set
    """
    # Only calculate on a subset of the batch to speed up validation
    subset_size = min(4, outputs.size(0))
    outputs = outputs[:subset_size]
    targets = targets[:subset_size]

    # Move tensors to CPU and convert to numpy
    outputs_np = outputs.detach().cpu().numpy()
    targets_np = targets.detach().cpu().numpy()

    psnr_sum = 0
    ssim_sum = 0

    # Calculate metrics for each image in batch
    for i in range(subset_size):
        output_img = outputs_np[i, 0]
        target_img = targets_np[i, 0]

        # Calculate PSNR
        psnr = peak_signal_noise_ratio(target_img, output_img, data_range=1.0)
        psnr_sum += psnr

        # Calculate SSIM
        ssim = structural_similarity(target_img, output_img, data_range=1.0)
        ssim_sum += ssim

    # Return average metrics
    return psnr_sum / subset_size, ssim_sum / subset_size


def log_images_to_wandb(inputs, outputs, targets, max_images=2):
    """
    Log input, output, and target images to Weights & Biases
    - Reduced to 2 images max for better performance
    """
    images = []
    num_images = min(max_images, inputs.size(0))

    for i in range(num_images):
        # Get images from batch and move to CPU
        input_img = inputs[i, 0].cpu().numpy()
        output_img = outputs[i, 0].detach().cpu().numpy()
        target_img = targets[i, 0].cpu().numpy()

        # Create wandb Image objects
        images.append(wandb.Image(
            input_img,
            caption=f"Input {i}",
            grouping=i
        ))

        images.append(wandb.Image(
            output_img,
            caption=f"Prediction {i}",
            grouping=i
        ))

        images.append(wandb.Image(
            target_img,
            caption=f"Ground Truth {i}",
            grouping=i
        ))

    # Log images to wandb
    wandb.log({"sample_images": images})


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                device, num_epochs, output_dir, save_every, early_stop_threshold,
                use_wandb=False, use_amp=False, clip_grad_norm=1.0,
                log_images_every=5, validation_metrics_every=5):
    """Train the model with optimized settings"""
    best_val_loss = float('inf')
    best_model_weights = None

    train_losses = []
    val_losses = []

    # Initialize mixed precision scaler if using AMP - forced on for better performance
    scaler = GradScaler() if (use_amp or device.type == 'cuda') else None

    # Initial wandb setup with reduced logging frequency
    if use_wandb:
        wandb.watch(model, log="all", log_freq=500)

        # Log model architecture as text
        wandb.log({
            "model_summary": str(model),
            "model_parameters": count_parameters(model),
            "model_size_mb": get_model_size_mb(model)
        })

    print(f"Starting training for {num_epochs} epochs...")
    print(f"Model has {count_parameters(model):,} parameters, size: {get_model_size_mb(model):.2f} MB")

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")

        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)

            # Zero the parameter gradients
            optimizer.zero_grad(set_to_none=True)  # More efficient than just zero_grad()

            # Forward pass with mixed precision (always enable for CUDA)
            if device.type == 'cuda' or use_amp:
                with autocast('cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                # Backward pass with gradient scaling
                scaler.scale(loss).backward()

                # Gradient clipping
                if clip_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)

                # Step optimizer and update scaler
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard forward/backward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                # Backward pass
                loss.backward()

                # Gradient clipping
                if clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)

                # Optimizer step
                optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = train_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        # Validation phase - simplified to run faster
        model.eval()
        val_loss = 0.0
        val_psnr = 0.0
        val_ssim = 0.0
        calculate_metrics_this_epoch = (epoch + 1) % validation_metrics_every == 0 or epoch == 0 or epoch == num_epochs - 1
        log_images_this_epoch = (epoch + 1) % log_images_every == 0 or epoch == 0 or epoch == num_epochs - 1

        progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Valid]")

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(progress_bar):
                inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)

                # Calculate image quality metrics only periodically
                if calculate_metrics_this_epoch:
                    psnr, ssim = calculate_metrics(outputs, targets)
                    val_psnr += psnr
                    val_ssim += ssim

                progress_bar.set_postfix(loss=f"{loss.item():.4f}")

                # Log sample images for the first batch periodically
                if use_wandb and log_images_this_epoch and batch_idx == 0:
                    log_images_to_wandb(inputs, outputs, targets)

        val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(val_loss)

        if calculate_metrics_this_epoch and val_loader.dataset:
            # Normalize metrics by the number of batches we calculated them on
            num_batches = len(val_loader)
            val_psnr = val_psnr / num_batches
            val_ssim = val_ssim / num_batches

        # Update learning rate with scheduler
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # Log based on whether we calculated metrics this epoch
        if calculate_metrics_this_epoch:
            print(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                  f"PSNR: {val_psnr:.2f}, SSIM: {val_ssim:.4f}, LR: {current_lr:.6f}")
        else:
            print(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                  f"LR: {current_lr:.6f}")

        # Log metrics to wandb
        if use_wandb:
            log_dict = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "learning_rate": current_lr
            }

            # Only add metrics if we calculated them
            if calculate_metrics_this_epoch:
                log_dict["val_psnr"] = val_psnr
                log_dict["val_ssim"] = val_ssim

            wandb.log(log_dict)

        # Save checkpoint periodically
        if (epoch + 1) % save_every == 0:
            checkpoint_path = os.path.join(output_dir, f'checkpoint_epoch_{epoch+1}.pth')
            save_model(
                model, optimizer, epoch, val_loss, output_dir,
                f'checkpoint_epoch_{epoch+1}.pth'
            )

            # Log model checkpoint to wandb
            if use_wandb:
                wandb.save(checkpoint_path)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_weights = copy.deepcopy(model.state_dict())
            best_model_path = os.path.join(output_dir, 'best_model.pth')
            save_model(
                model, optimizer, epoch, val_loss, output_dir,
                'best_model.pth'
            )
            print(f"New best model with validation loss: {val_loss:.4f}")

            # Log best model to wandb
            if use_wandb:
                wandb.run.summary["best_val_loss"] = best_val_loss
                if calculate_metrics_this_epoch:
                    wandb.run.summary["best_val_psnr"] = val_psnr
                    wandb.run.summary["best_val_ssim"] = val_ssim
                wandb.run.summary["best_epoch"] = epoch + 1
                wandb.save(best_model_path)

        # Early stopping check
        if val_loss < early_stop_threshold:
            print(f"Early stopping at epoch {epoch+1} with validation loss {val_loss:.4f} < {early_stop_threshold}")
            break

    # Plot training progress
    plot_losses(train_losses, val_losses, output_dir)

    # Load best model weights
    model.load_state_dict(best_model_weights)
    print(f"Training completed. Best validation loss: {best_val_loss:.4f}")

    # Finish wandb run
    if use_wandb:
        wandb.finish()

    return model, best_val_loss


def main():
    """Main function for training the glare removal model"""
    args = parse_args()

    # Environmental variables for reproducibility
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

    # Set random seed for reproducibility
    set_seed(args.seed)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Ensure deterministic algorithms are used
    if hasattr(torch, 'use_deterministic_algorithms'):
        torch.use_deterministic_algorithms(True)
    elif hasattr(torch, 'set_deterministic'):
        torch.set_deterministic(True)

    # Disable cuDNN benchmark mode for reproducibility
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    # Create data generators for worker seeding
    g_train = torch.Generator()
    g_train.manual_seed(args.seed)
    g_val = torch.Generator()
    g_val.manual_seed(args.seed + 1)  # Different seed for validation

    # Create optimized dataloaders with worker seeding
    train_loader, val_loader = create_optimized_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        val_split=args.val_split,
        num_workers=args.num_workers,
        seed=args.seed,
        image_size=args.image_size,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=args.persistent_workers,
        worker_init_fn=seed_worker,
        generator=g_train,
        val_generator=g_val
    )
    print(f"Training samples: {len(train_loader.dataset)}, Validation samples: {len(val_loader.dataset)}")

    # Initialize Weights & Biases if requested
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config={
                "model": args.model,
                "batch_size": args.batch_size,
                "learning_rate": args.lr,
                "epochs": args.epochs,
                "weight_decay": args.weight_decay,
                "early_stop_threshold": args.early_stop,
                "use_amp": args.use_amp,
                "clip_grad_norm": args.clip_grad_norm,
                "seed": args.seed,
                "image_size": args.image_size
            }
        )

    # Create model based on the selected architecture
    if args.model == 'enhanced':
        model = EnhancedUNet(in_channels=1, out_channels=1).to(device)
        print("Using enhanced U-Net architecture with residual blocks and attention gates")
    elif args.model == 'optimized':
        model = OptimizedUNet(in_channels=1, out_channels=1).to(device)
        print("Using optimized U-Net architecture for better speed/accuracy tradeoff")
    else:
        model = LightweightUNet(in_channels=1, out_channels=1).to(device)
        print("Using basic lightweight U-Net architecture")

    # Define loss function and optimizer with more modern settings
    criterion = nn.L1Loss()
    optimizer = torch.optim.AdamW(  # AdamW is often better than Adam
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-08
    )

    # Simplified scheduler - just ReduceLROnPlateau
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # Train the model
    trained_model, best_val_loss = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=args.epochs,
        output_dir=args.output_dir,
        save_every=args.save_every,
        early_stop_threshold=args.early_stop,
        use_wandb=args.use_wandb,
        use_amp=args.use_amp,
        clip_grad_norm=args.clip_grad_norm,
        log_images_every=args.log_images_every,
        validation_metrics_every=args.validation_metrics_every
    )

    # Save final model
    save_model(
        trained_model, optimizer, args.epochs, best_val_loss,
        args.output_dir, 'final_model.pth'
    )

    # Save model in smaller format (just weights)
    torch.save(trained_model.state_dict(), os.path.join(args.output_dir, 'model_weights.pth'))
    print(f"Model weights saved to {os.path.join(args.output_dir, 'model_weights.pth')}")

    # Print model size information
    model_size_mb = get_model_size_mb(trained_model)
    print(f"Final model size: {model_size_mb:.2f} MB")
    if model_size_mb > 4.0:
        print("Warning: Model exceeds 4MB target size. Consider applying quantization or pruning.")


if __name__ == '__main__':
    main()