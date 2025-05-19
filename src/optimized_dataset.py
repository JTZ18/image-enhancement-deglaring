import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random
from PIL import Image
from functools import partial


class OptimizedGlareRemovalDataset(Dataset):
    """
    Optimized dataset for glare removal task
    - Pre-processes images during initialization
    - Caches processed images in memory for faster access
    - Uses smaller image size
    - Simplified augmentations
    """
    def __init__(self, image_paths, transform=None, seed=None, image_size=256, cache_images=True):
        """
        Initialize the dataset with optimized memory caching

        Args:
            image_paths (list): List of image file paths
            transform (albumentations.Compose, optional): Transformations to apply
            seed (int, optional): Random seed for reproducibility
            image_size (int): Size to resize images to (smaller = faster training)
            cache_images (bool): Whether to cache processed images in memory
        """
        self.image_paths = image_paths
        self.transform = transform
        self.seed = seed
        self.image_size = image_size
        self.cache_images = cache_images

        # Sort image paths to ensure consistent ordering
        self.image_paths.sort()

        # Pre-process and cache images for faster training
        self.cached_images = {}
        if self.cache_images:
            self._cache_images()

    def _cache_images(self):
        """Pre-process and cache images for faster access during training"""
        print(f"Pre-processing and caching {len(self.image_paths)} images...")

        for i, path in enumerate(self.image_paths):
            if i % 100 == 0:
                print(f"Processed {i}/{len(self.image_paths)} images")

            # Load the image
            img = cv2.imread(path)
            if img is None:
                print(f"Warning: Could not load image {path}, skipping")
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Split the concatenated image
            width = img.shape[1]
            third = width // 3

            ground_truth = img[:, :third, :]
            glared_image = img[:, third:2*third, :]

            # Convert to grayscale
            ground_truth_gray = cv2.cvtColor(ground_truth, cv2.COLOR_RGB2GRAY)
            glared_image_gray = cv2.cvtColor(glared_image, cv2.COLOR_RGB2GRAY)

            # Resize to reduce memory usage and speed up training
            glared_image_gray = cv2.resize(glared_image_gray, (self.image_size, self.image_size))
            ground_truth_gray = cv2.resize(ground_truth_gray, (self.image_size, self.image_size))

            # Normalize to [0, 1]
            glared_image_gray = glared_image_gray.astype(np.float32) / 255.0
            ground_truth_gray = ground_truth_gray.astype(np.float32) / 255.0

            # Store in cache
            self.cached_images[i] = (glared_image_gray, ground_truth_gray)

        print(f"Completed caching {len(self.cached_images)} images")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        # Set seed for reproducible augmentations
        if self.seed is not None:
            seed_for_augmentation = self.seed + index
            random.seed(seed_for_augmentation)
            np.random.seed(seed_for_augmentation)
            torch.manual_seed(seed_for_augmentation)

        # Use cached images if available
        if self.cache_images and index in self.cached_images:
            glared_image_gray, ground_truth_gray = self.cached_images[index]
        else:
            # Load and process image if not cached
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

            # Resize to reduce memory usage and speed up training
            glared_image_gray = cv2.resize(glared_image_gray, (self.image_size, self.image_size))
            ground_truth_gray = cv2.resize(ground_truth_gray, (self.image_size, self.image_size))

            # Normalize to [0, 1]
            glared_image_gray = glared_image_gray.astype(np.float32) / 255.0
            ground_truth_gray = ground_truth_gray.astype(np.float32) / 255.0

        # Apply transformations
        if self.transform:
            # Make sure masks are correctly transformed with the same augmentations
            augmented = self.transform(image=glared_image_gray, mask=ground_truth_gray)
            glared_image_tensor = augmented['image']  # Already a tensor from ToTensorV2
            ground_truth_tensor = augmented['mask']  # Already a tensor from ToTensorV2

            # Ensure channel dimension exists
            if len(ground_truth_tensor.shape) == 2:
                ground_truth_tensor = ground_truth_tensor.unsqueeze(0)
        else:
            # Convert to tensor
            glared_image_tensor = torch.from_numpy(glared_image_gray).unsqueeze(0)  # Add channel dimension
            ground_truth_tensor = torch.from_numpy(ground_truth_gray).unsqueeze(0)  # Add channel dimension

        return glared_image_tensor, ground_truth_tensor


def get_optimized_transformations(image_size=256, seed=42):
    """
    Get simplified data augmentation transformations for faster training

    Args:
        image_size (int): Size of images for training
        seed (int): Random seed for reproducibility

    Returns:
        tuple: (train_transform, val_transform) for training and validation datasets
    """
    # Simplified training transforms for better performance
    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        # Reduced set of augmentations for faster processing
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.8),
            A.GaussNoise(p=0.2),
        ], p=0.5),
        ToTensorV2(),
    ], seed=seed)

    # Minimal validation transforms
    val_transform = A.Compose([
        ToTensorV2(),
    ], seed=seed)

    return train_transform, val_transform


def create_optimized_dataloaders(data_dir, batch_size=64, val_split=0.2, num_workers=8,
                           seed=None, image_size=256, cache_images=False,
                           prefetch_factor=2, persistent_workers=True,
                           worker_init_fn=None, generator=None, val_generator=None):
    """
    Create optimized training and validation dataloaders

    Args:
        data_dir (str): Directory containing the images
        batch_size (int): Batch size for dataloaders (increased for better GPU utilization)
        val_split (float): Validation split ratio (0.0 to 1.0)
        num_workers (int): Number of worker threads for dataloaders
        seed (int, optional): Seed for reproducible dataset splitting
        image_size (int): Size to resize images to
        cache_images (bool): Whether to cache processed images in memory
        prefetch_factor (int): Number of batches to prefetch per worker
        persistent_workers (bool): Keep workers alive between epochs

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
    train_transform, val_transform = get_optimized_transformations(image_size, seed=seed)

    # Create optimized datasets
    train_dataset = OptimizedGlareRemovalDataset(
        train_paths,
        transform=train_transform,
        seed=seed,
        image_size=image_size,
        cache_images=cache_images
    )

    val_dataset = OptimizedGlareRemovalDataset(
        val_paths,
        transform=val_transform,
        seed=seed,
        image_size=image_size,
        cache_images=cache_images
    )

    # Define worker initialization function for reproducibility if not provided
    if worker_init_fn is None:
        def worker_init_fn(worker_id):
            # Each worker needs a different seed but deterministically derived from the global seed
            if seed is not None:
                worker_seed = seed + worker_id
            else:
                worker_seed = torch.initial_seed() % 2**32

            # Set random seeds for this worker
            np.random.seed(worker_seed)
            random.seed(worker_seed)
            torch.manual_seed(worker_seed)

    # Create generators for dataloaders if not provided
    if generator is None:
        generator = torch.Generator()
        if seed is not None:
            generator.manual_seed(seed)

    if val_generator is None:
        val_generator = torch.Generator()
        if seed is not None:
            val_generator.manual_seed(seed + 10000)  # Use a different seed for validation

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        generator=generator,
        drop_last=True,  # For consistent batch sizes
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers if num_workers > 0 else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=max(2, num_workers//2),  # Use fewer workers for validation
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        generator=val_generator,
        persistent_workers=persistent_workers if num_workers > 0 else False
    )

    return train_loader, val_loader