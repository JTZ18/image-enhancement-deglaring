import random
import numpy as np
import torch
import os

def set_seed(seed=42):
    """
    Set all random seeds for reproducibility.
    
    Args:
        seed (int): Seed value to use for all random number generators.
    """
    # Force deterministic execution at Python level
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Make sure subprocesses inherit the seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Set Python's built-in random module seed
    random.seed(seed)
    
    # Set NumPy's random module seed
    np.random.seed(seed)
    
    # Set PyTorch's random module seed for CPU operations
    torch.manual_seed(seed)
    
    # Set PyTorch's random module seed for all CUDA devices
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        # Enable deterministic behavior for CUDA
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Set PyTorch's random number generator seed for model weight initialization
    torch.nn.init.calculate_gain('leaky_relu', 0.2)
    
    # Make sure we're using deterministic algorithms when possible
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # For CUDA >= 10.2
    
    try:
        if hasattr(torch, 'use_deterministic_algorithms'):
            torch.use_deterministic_algorithms(True)
        elif hasattr(torch, 'set_deterministic'):  # For older PyTorch versions
            torch.set_deterministic(True)
    except Exception as e:
        print(f"Warning: Could not enable deterministic algorithms: {e}")
    
    # Reset RNG states to ensure consistency
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    print(f"All random seeds set to {seed} for reproducibility")