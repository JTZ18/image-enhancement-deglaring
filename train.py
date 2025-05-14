import os
import argparse
import copy
import torch
import torch.nn as nn
from torch.utils.data import random_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

# Import from src modules
from src.dataset import create_dataloaders
from src.model import LightweightUNet, count_parameters, get_model_size_mb


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train glare removal model')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing the dataset')
    parser.add_argument('--output_dir', type=str, default='./models', help='Directory to save model')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--val_split', type=float, default=0.2, help='Validation split ratio')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of dataloader workers')
    parser.add_argument('--save_every', type=int, default=10, help='Save checkpoint every N epochs')
    parser.add_argument('--early_stop', type=float, default=0.06, help='Early stopping loss threshold')
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


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                device, num_epochs, output_dir, save_every, early_stop_threshold):
    """Train the model and keep track of progress"""
    best_val_loss = float('inf')
    best_model_weights = None

    train_losses = []
    val_losses = []

    print(f"Starting training for {num_epochs} epochs...")
    print(f"Model has {count_parameters(model):,} parameters, size: {get_model_size_mb(model):.2f} MB")

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")

        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = train_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0
        progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Valid]")

        with torch.no_grad():
            for inputs, targets in progress_bar:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(val_loss)

        # Update learning rate
        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Save checkpoint periodically
        if (epoch + 1) % save_every == 0:
            save_model(
                model, optimizer, epoch, val_loss, output_dir,
                f'checkpoint_epoch_{epoch+1}.pth'
            )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_weights = copy.deepcopy(model.state_dict())
            save_model(
                model, optimizer, epoch, val_loss, output_dir,
                'best_model.pth'
            )
            print(f"New best model with validation loss: {val_loss:.4f}")

        # Early stopping check
        if val_loss < early_stop_threshold:
            print(f"Early stopping at epoch {epoch+1} with validation loss {val_loss:.4f} < {early_stop_threshold}")
            break

    # Plot training progress
    plot_losses(train_losses, val_losses, output_dir)

    # Load best model weights
    model.load_state_dict(best_model_weights)
    print(f"Training completed. Best validation loss: {best_val_loss:.4f}")

    return model, best_val_loss


def main():
    """Main function for training the glare removal model"""
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        val_split=args.val_split,
        num_workers=args.num_workers
    )
    print(f"Training samples: {len(train_loader.dataset)}, Validation samples: {len(val_loader.dataset)}")

    # Create model
    model = LightweightUNet(in_channels=1, out_channels=1).to(device)

    # Define loss function and optimizer
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
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
        early_stop_threshold=args.early_stop
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