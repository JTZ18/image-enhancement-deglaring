import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random

from .optimized_dataset import create_optimized_dataloaders


class GlareRemovalDataset(Dataset):
    """
    Dataset for glare removal task handling concatenated images in format 
    [Ground Truth, Glared Image, Glare Mask]
    """
    def __init__(self, image_paths, transform=None, seed=None):
        """
        Initialize the dataset
        
        Args:
            image_paths (list): List of image file paths
            transform (albumentations.Compose, optional): Transformations to apply
            seed (int, optional): Random seed for reproducibility
        """
        self.image_paths = image_paths
        self.transform = transform
        self.seed = seed
        
        # Sort image paths to ensure consistent ordering
        self.image_paths.sort()
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        # Set seed based on fixed seed and index for reproducible augmentations
        if self.seed is not None:
            # Use different seed for each image but derived deterministically from base seed
            # This ensures the same image gets the same augmentations in multiple runs
            seed_for_augmentation = self.seed + index
            random.seed(seed_for_augmentation)
            np.random.seed(seed_for_augmentation)
            torch.manual_seed(seed_for_augmentation)
            
        image_path = self.image_paths[index]
        
        # Load the image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Split the concatenated image
        width = img.shape[1]
        third = width // 3
        
        ground_truth = img[:, :third, :]
        glared_image = img[:, third:2*third, :]
        
        # Convert to grayscale
        ground_truth_gray = cv2.cvtColor(ground_truth, cv2.COLOR_RGB2GRAY)
        glared_image_gray = cv2.cvtColor(glared_image, cv2.COLOR_RGB2GRAY)
        
        # Apply transformations
        if self.transform:
            # Make sure masks are correctly transformed with the same augmentations
            # Force deterministic behavior for transformations
            augmented = self.transform(image=glared_image_gray, mask=ground_truth_gray)
            glared_image_tensor = augmented['image']  # Already a tensor from ToTensorV2
            ground_truth_tensor = augmented['mask']  # Already a tensor from ToTensorV2
            
            # Ensure channel dimension exists and both tensors have same dimensions [C,H,W]
            if len(ground_truth_tensor.shape) == 2:
                ground_truth_tensor = ground_truth_tensor.unsqueeze(0)
        else:
            # Normalize to [0, 1]
            glared_image_gray = glared_image_gray.astype(np.float32) / 255.0
            ground_truth_gray = ground_truth_gray.astype(np.float32) / 255.0
            
            # Convert to tensor
            glared_image_tensor = torch.from_numpy(glared_image_gray).unsqueeze(0)  # Add channel dimension
            ground_truth_tensor = torch.from_numpy(ground_truth_gray).unsqueeze(0)  # Add channel dimension
            
        return glared_image_tensor, ground_truth_tensor


def get_transformations():
    """
    Get data augmentation transformations for training and validation
    
    Returns:
        tuple: (train_transform, val_transform) for training and validation datasets
    """
    train_transform = A.Compose([
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),  # Add vertical flip
        # Replace ShiftScaleRotate with Affine as recommended
        A.Affine(scale=(0.9, 1.1), translate_percent=0.0625, rotate=(-15, 15), p=0.5),
        A.OneOf([
            # Using correct GaussNoise parameters
            A.GaussNoise(p=0.5),  # Use default parameters
            A.GaussianBlur(blur_limit=3, p=0.5),
        ], p=0.5),
        # Add glare-specific augmentations
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.RandomGamma(gamma_limit=(80, 120), p=0.5),  # Helps with glare-like effects
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.5),  # Enhances contrast locally
        ], p=0.5),
        A.Resize(512, 512),
        A.Normalize(mean=0.5, std=0.5),
        ToTensorV2(),
    ])
    
    val_transform = A.Compose([
        A.Resize(512, 512),
        A.Normalize(mean=0.5, std=0.5),
        ToTensorV2(),
    ])
    
    return train_transform, val_transform


def create_dataloaders(data_dir, batch_size=32, val_split=0.2, num_workers=4, seed=None):
    """
    Create training and validation dataloaders
    
    Args:
        data_dir (str): Directory containing the images
        batch_size (int): Batch size for dataloaders
        val_split (float): Validation split ratio (0.0 to 1.0)
        num_workers (int): Number of worker threads for dataloaders
        seed (int, optional): Seed for reproducible dataset splitting
        
    Returns:
        tuple: (train_loader, val_loader) DataLoader objects
    """
    # Get all image file paths
    image_paths = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(root, file))
    
    # Sort paths to ensure consistent order before shuffling
    image_paths.sort()
    
    # Shuffle and split into train/val with reproducible seed if provided
    if seed is not None:
        rng = np.random.RandomState(seed)
        rng.shuffle(image_paths)
    else:
        np.random.shuffle(image_paths)
        
    split_idx = int(len(image_paths) * (1 - val_split))
    train_paths = image_paths[:split_idx]
    val_paths = image_paths[split_idx:]
    
    # Get transformations
    train_transform, val_transform = get_transformations()
    
    # Create datasets
    train_dataset = GlareRemovalDataset(train_paths, transform=train_transform, seed=seed)
    val_dataset = GlareRemovalDataset(val_paths, transform=val_transform, seed=seed)
    
    # Define worker initialization function for reproducibility
    def worker_init_fn(worker_id):
        # Each worker needs a different seed but deterministically derived from the global seed
        if seed is not None:
            worker_seed = seed + worker_id
        else:
            worker_seed = torch.initial_seed() % 2**32
        
        # Set all random seeds for this worker
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)
    
    # Create dataloaders with deterministic behavior
    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)
    
    # Force deterministic behavior for data loading
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        generator=generator,
        drop_last=True  # For consistent batch sizes
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn
    )
    
    return train_loader, val_loader