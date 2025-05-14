import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2


class GlareRemovalDataset(Dataset):
    """
    Dataset for glare removal task handling concatenated images in format 
    [Ground Truth, Glared Image, Glare Mask]
    """
    def __init__(self, image_paths, transform=None):
        """
        Initialize the dataset
        
        Args:
            image_paths (list): List of image file paths
            transform (albumentations.Compose, optional): Transformations to apply
        """
        self.image_paths = image_paths
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
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
        # Replace ShiftScaleRotate with Affine as recommended
        A.Affine(scale=(0.9, 1.1), translate_percent=0.0625, rotate=(-15, 15), p=0.5),
        A.OneOf([
            # Fix GaussNoise parameters
            A.GaussNoise(mean=0, var_limit=10.0, per_channel=True, p=0.5),
            A.GaussianBlur(blur_limit=3, p=0.5),
        ], p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
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


def create_dataloaders(data_dir, batch_size=32, val_split=0.2, num_workers=4):
    """
    Create training and validation dataloaders
    
    Args:
        data_dir (str): Directory containing the images
        batch_size (int): Batch size for dataloaders
        val_split (float): Validation split ratio (0.0 to 1.0)
        num_workers (int): Number of worker threads for dataloaders
        
    Returns:
        tuple: (train_loader, val_loader) DataLoader objects
    """
    # Get all image file paths
    image_paths = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(root, file))
    
    # Shuffle and split into train/val
    np.random.shuffle(image_paths)
    split_idx = int(len(image_paths) * (1 - val_split))
    train_paths = image_paths[:split_idx]
    val_paths = image_paths[split_idx:]
    
    # Get transformations
    train_transform, val_transform = get_transformations()
    
    # Create datasets
    train_dataset = GlareRemovalDataset(train_paths, transform=train_transform)
    val_dataset = GlareRemovalDataset(val_paths, transform=val_transform)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader