import numpy as np
import cv2
from PIL import Image
import torch

def preprocess_data(image_path, image_size=512):
    """
    Preprocess data from concatenated images in format [Ground Truth, Glared Image, Glare Mask]
    
    Args:
        image_path (str): Path to the image file
        image_size (int): Target image size
        
    Returns:
        tuple: Pair of tensors (glared_image, ground_truth)
    """
    # Load the image
    img = Image.open(image_path)
    img = np.array(img)
    
    # Split the concatenated image
    width = img.shape[1]
    third = width // 3
    
    ground_truth = img[:, :third, :]          # Ground truth image
    glared_image = img[:, third:2*third, :]   # Glared image
    # glare_mask = img[:, 2*third:, :]        # Glare mask (not needed)
    
    # Convert to grayscale
    if ground_truth.shape[2] == 4:  # RGBA
        # Convert to grayscale using luminance formula
        ground_truth_gray = 0.299 * ground_truth[:,:,0] + 0.587 * ground_truth[:,:,1] + 0.114 * ground_truth[:,:,2]
        glared_image_gray = 0.299 * glared_image[:,:,0] + 0.587 * glared_image[:,:,1] + 0.114 * glared_image[:,:,2]
    else:  # RGB
        ground_truth_gray = 0.299 * ground_truth[:,:,0] + 0.587 * ground_truth[:,:,1] + 0.114 * ground_truth[:,:,2]
        glared_image_gray = 0.299 * glared_image[:,:,0] + 0.587 * glared_image[:,:,1] + 0.114 * glared_image[:,:,2]
    
    # Resize to 512x512 if needed
    if ground_truth_gray.shape[0] != image_size or ground_truth_gray.shape[1] != image_size:
        ground_truth_gray = cv2.resize(ground_truth_gray, (image_size, image_size))
        glared_image_gray = cv2.resize(glared_image_gray, (image_size, image_size))
    
    # Normalize to [0, 1]
    ground_truth_gray = ground_truth_gray.astype(np.float32) / 255.0
    glared_image_gray = glared_image_gray.astype(np.float32) / 255.0
    
    # Convert to tensor
    ground_truth_tensor = torch.from_numpy(ground_truth_gray).unsqueeze(0)  # Add channel dimension
    glared_image_tensor = torch.from_numpy(glared_image_gray).unsqueeze(0)  # Add channel dimension
    
    return glared_image_tensor, ground_truth_tensor


def preprocess_inference(image_path, image_size=512):
    """
    Preprocess a single image for inference
    
    Args:
        image_path (str): Path to the image file
        image_size (int): Target image size
        
    Returns:
        torch.Tensor: Preprocessed image tensor
    """
    # Load the image
    img = Image.open(image_path)
    img = np.array(img)
    
    # Convert to grayscale
    if len(img.shape) == 3 and img.shape[2] in [3, 4]:
        # RGB or RGBA image
        if img.shape[2] == 4:  # RGBA
            img_gray = 0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]
        else:  # RGB
            img_gray = 0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]
    else:
        # Already grayscale
        img_gray = img
    
    # Resize to image_size x image_size
    if img_gray.shape[0] != image_size or img_gray.shape[1] != image_size:
        img_gray = cv2.resize(img_gray, (image_size, image_size))
    
    # Normalize to [0, 1]
    img_gray = img_gray.astype(np.float32) / 255.0
    
    # Convert to tensor
    img_tensor = torch.from_numpy(img_gray).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    
    return img_tensor


def postprocess_output(output_tensor):
    """
    Convert model output tensor to image
    
    Args:
        output_tensor (torch.Tensor): Model output tensor
        
    Returns:
        numpy.ndarray: Processed image as numpy array
    """
    # Ensure the tensor is on CPU and convert to numpy
    output_np = output_tensor.detach().cpu().squeeze().numpy()
    
    # Clip values to [0, 1] range
    output_np = np.clip(output_np, 0, 1)
    
    # Scale to [0, 255] and convert to uint8
    output_image = (output_np * 255).astype(np.uint8)
    
    return output_image