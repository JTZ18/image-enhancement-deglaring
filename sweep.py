#!/usr/bin/env python3
import os
import argparse
import wandb
import torch
import random
import numpy as np
from functools import partial
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

# Import modules
from src.dataset import create_optimized_dataloaders
from src.model import LightweightUNet, EnhancedUNet, count_parameters, get_model_size_mb
from src.optimized_model import OptimizedUNet
from src.utils import set_seed
from optimized_train import train_model, save_model, seed_worker


def parse_args():
    """Parse command line arguments for the sweep configuration"""
    parser = argparse.ArgumentParser(description='Run a hyperparameter sweep for glare removal model')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing the dataset')
    parser.add_argument('--output_dir', type=str, default='./models', help='Directory to save model checkpoints')
    parser.add_argument('--sweep_project', type=str, default='image-deglaring-sweep', help='Weights & Biases project name for sweep')
    parser.add_argument('--sweep_entity', type=str, default=None, help='Weights & Biases entity (team) name')
    parser.add_argument('--sweep_count', type=int, default=20, help='Number of sweep runs to perform')
    parser.add_argument('--val_split', type=float, default=0.2, help='Validation split ratio')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of dataloader workers')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--max_epochs', type=int, default=50, help='Maximum epochs for each sweep run')
    parser.add_argument('--early_stop_patience', type=int, default=10, help='Early stopping patience in epochs')
    parser.add_argument('--prefetch_factor', type=int, default=2, help='Prefetch factor for dataloader')
    parser.add_argument('--persistent_workers', action='store_true', help='Keep workers alive between epochs')
    return parser.parse_args()


def setup_sweep_config():
    """Define the hyperparameter sweep configuration with Bayesian optimization"""
    sweep_config = {
        'method': 'bayes',  # Bayesian optimization
        'metric': {
            'name': 'val_loss',
            'goal': 'minimize'
        },
        'early_terminate': {
            'type': 'hyperband',
            'min_iter': 10,
            's': 2,
        },
        'parameters': {
            # Model architecture choice
            # 'model': {
            #     'values': ['basic', 'enhanced', 'optimized']
            # },

            # Training parameters
            'batch_size': {
                'values': [4, 8, 16, 32]
            },
            'learning_rate': {
                'distribution': 'log_uniform_values',
                'min': 1e-4,
                'max': 1e-2
            },
            # 'optimizer': {
            #     'values': ['adam', 'adamw']
            # },
            'weight_decay': {
                'distribution': 'log_uniform_values',
                'min': 1e-6,
                'max': 1e-3
            },


            # Fixed parameters
            'use_amp': {'value': True},  # Always use mixed precision for better performance
            'clip_grad_norm': {'value': 1.0},  # Gradient clipping
            'save_every': {'value': 10},  # Save every n epochs
            'log_images_every': {'value': 5},  # Log images every n epochs
            'validation_metrics_every': {'value': 5},  # Calculate validation metrics every n epochs
            'image_size': {'value': 512},  # Image size
            'model': {'value': 'basic'},
            'optimizer': {'value': 'adamw'}
        }
    }

    return sweep_config


def train_sweep():
    """Training function for a single sweep run"""
    # Initialize wandb for this run
    wandb.init()

    # Access sweep configuration
    config = wandb.config

    # Environment setup for reproducibility
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Create data generators for worker seeding
    g_train = torch.Generator()
    g_train.manual_seed(args.seed)
    g_val = torch.Generator()
    g_val.manual_seed(args.seed + 1)

    # Create output directory for this run
    run_output_dir = os.path.join(args.output_dir, f"sweep_{wandb.run.id}")
    os.makedirs(run_output_dir, exist_ok=True)

    # Create dataloaders
    train_loader, val_loader = create_optimized_dataloaders(
        args.data_dir,
        batch_size=config.batch_size,
        val_split=args.val_split,
        num_workers=args.num_workers,
        seed=args.seed,
        image_size=config.image_size,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=args.persistent_workers,
        worker_init_fn=seed_worker,
        generator=g_train,
        val_generator=g_val
    )

    logger.info(f"Created dataloaders - Training samples: {len(train_loader.dataset)}, Validation samples: {len(val_loader.dataset)}")

    # Create model based on sweep parameter 'model'
    if config.model == 'enhanced':
        model = EnhancedUNet(in_channels=1, out_channels=1).to(device)
        logger.info("Using enhanced U-Net architecture")
    elif config.model == 'optimized':
        model = OptimizedUNet(in_channels=1, out_channels=1).to(device)
        logger.info("Using optimized U-Net architecture")
    else:  # basic/lightweight
        model = LightweightUNet(in_channels=1, out_channels=1).to(device)
        logger.info("Using basic lightweight U-Net architecture")

    # Calculate and log model parameters and size
    model_params = count_parameters(model)
    model_size = get_model_size_mb(model)
    logger.info(f"Model has {model_params:,} parameters, size: {model_size:.2f} MB")

    # Define loss function
    criterion = torch.nn.L1Loss()

    # Select optimizer based on sweep parameter 'optimizer'
    if config.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        logger.info("Using AdamW optimizer")
    else:  # adam
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        logger.info("Using Adam optimizer")

    # Scheduler (ReduceLROnPlateau)
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
        num_epochs=args.max_epochs,
        output_dir=run_output_dir,
        save_every=config.save_every,
        patience=args.early_stop_patience,
        use_wandb=True,
        use_amp=config.use_amp,
        clip_grad_norm=config.clip_grad_norm,
        log_images_every=config.log_images_every,
        validation_metrics_every=config.validation_metrics_every
    )

    # Save final model
    save_model(
        trained_model, optimizer, args.max_epochs, best_val_loss,
        run_output_dir, 'final_model.pth'
    )

    # Save model weights only (smaller file)
    weights_path = os.path.join(run_output_dir, 'model_weights.pth')
    torch.save(trained_model.state_dict(), weights_path)

    # Log final model size
    model_size_mb = get_model_size_mb(trained_model)
    wandb.log({"final_model_size_mb": model_size_mb})

    logger.info(f"Sweep run completed. Best validation loss: {best_val_loss:.4f}")
    logger.info(f"Model size: {model_size_mb:.2f} MB")

    if model_size_mb > 4.0:
        logger.warning("Model exceeds 4MB target size")

    # Return final metric value
    return best_val_loss


def main():
    """Main function to run hyperparameter sweep"""
    global args
    args = parse_args()

    # Setup sweep configuration
    sweep_config = setup_sweep_config()

    # Initialize sweep
    sweep_id = wandb.sweep(
        sweep_config,
        project=args.sweep_project,
        entity=args.sweep_entity
    )

    logger.info(f"Created sweep with ID: {sweep_id}")
    logger.info(f"Running {args.sweep_count} sweep iterations")

    # Run the sweep agent
    wandb.agent(sweep_id, function=train_sweep, count=args.sweep_count)

    logger.info("Sweep completed")


if __name__ == "__main__":
    main()