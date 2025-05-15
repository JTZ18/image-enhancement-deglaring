import os
import torch
import numpy as np
import random
from src.utils import set_seed

def test_reproducibility():
    """
    Test that setting seeds properly ensures reproducible results.
    
    This test creates two identical models, seeds the RNGs properly,
    and ensures they produce identical outputs for the same input.
    """
    # Environment variables for CUDA reproducibility
    os.environ["PYTHONHASHSEED"] = "42"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    
    # Set seed via our utility function
    set_seed(42)
    
    # Create test data
    input_tensor = torch.randn(4, 1, 64, 64)
    
    # Create identical models
    from src.optimized_model import OptimizedUNet
    model1 = OptimizedUNet(in_channels=1, out_channels=1)
    model2 = OptimizedUNet(in_channels=1, out_channels=1)
    
    # Ensure deterministic algorithms are used
    if hasattr(torch, 'use_deterministic_algorithms'):
        torch.use_deterministic_algorithms(True)
    
    # Get outputs from both models
    with torch.no_grad():
        output1 = model1(input_tensor)
        output2 = model2(input_tensor)
    
    # Check if outputs are identical
    is_identical = torch.allclose(output1, output2)
    
    print(f"Outputs are identical: {is_identical}")
    if not is_identical:
        print(f"Max difference: {(output1 - output2).abs().max().item()}")
    
    return is_identical

if __name__ == "__main__":
    if test_reproducibility():
        print("✅ Reproducibility test passed: Deterministic behavior confirmed")
    else:
        print("❌ Reproducibility test failed: Non-deterministic behavior detected")