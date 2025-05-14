import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


class LightweightUNet(nn.Module):
    """
    Lightweight U-Net architecture for image de-glaring
    Designed for grayscale images with a target parameter count under 1M
    """
    def __init__(self, in_channels=1, out_channels=1):
        """
        Initialize the lightweight U-Net model
        
        Args:
            in_channels (int): Number of input channels (1 for grayscale)
            out_channels (int): Number of output channels (1 for grayscale)
        """
        super().__init__()
        
        # Encoder (downsampling) - reduced channel counts to meet 4MB target
        self.enc1 = self._block(in_channels, 8, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc2 = self._block(8, 16, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc3 = self._block(16, 32, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc4 = self._block(32, 64, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Bottleneck
        self.bottleneck = self._block(64, 128, name="bottleneck")
        
        # Decoder (upsampling)
        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec4 = self._block(128, 64, name="dec4")  # 128 = 64 + 64 (skip connection)
        self.upconv3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec3 = self._block(64, 32, name="dec3")   # 64 = 32 + 32 (skip connection)
        self.upconv2 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec2 = self._block(32, 16, name="dec2")    # 32 = 16 + 16 (skip connection)
        self.upconv1 = nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2)
        self.dec1 = self._block(16, 8, name="dec1")    # 16 = 8 + 8 (skip connection)
        
        # Output layer
        self.output = nn.Conv2d(8, out_channels, kernel_size=1)
    
    def _block(self, in_channels, features, name):
        """
        Create a convolutional block for the U-Net
        
        Args:
            in_channels (int): Number of input channels
            features (int): Number of output channels
            name (str): Name of the block
            
        Returns:
            nn.Sequential: Sequential container of layers
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, C, H, W]
            
        Returns:
            torch.Tensor: Output tensor of shape [B, C, H, W]
        """
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        enc4 = self.enc4(self.pool3(enc3))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool4(enc4))
        
        # Decoder with skip connections
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.dec4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.dec1(dec1)
        
        return self.output(dec1)


def count_parameters(model):
    """
    Count the number of trainable parameters in a model
    
    Args:
        model (nn.Module): PyTorch model
        
    Returns:
        int: Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size_mb(model):
    """
    Calculate the size of a model in MB
    
    Args:
        model (nn.Module): PyTorch model
        
    Returns:
        float: Model size in MB
    """
    # Get model state dict
    state_dict = model.state_dict()
    
    # Calculate size in bytes
    size_bytes = sum(v.element_size() * v.nelement() for v in state_dict.values())
    
    # Convert to MB
    size_mb = size_bytes / (1024 * 1024)
    
    return size_mb


def quantize_model(model, calibration_dataloader=None):
    """
    Quantize model to reduce size
    
    Args:
        model (nn.Module): Model to quantize
        calibration_dataloader (DataLoader, optional): DataLoader for calibration
        
    Returns:
        nn.Module: Quantized model
    """
    # Prepare model for quantization
    model.eval()
    
    # TODO: Implement proper quantization logic depending on PyTorch version
    # This is a simplified implementation - proper implementation requires PyTorch 1.3+
    
    # For now, just return the original model
    print("Warning: Quantization not fully implemented - returning original model")
    return model


def prune_model(model, amount=0.3):
    """
    Prune model to reduce size by removing weights
    
    Args:
        model (nn.Module): Model to prune
        amount (float): Amount of weights to prune (0.0 to 1.0)
        
    Returns:
        nn.Module: Pruned model
    """
    # TODO: Implement proper pruning logic depending on PyTorch version
    # This requires PyTorch's utils.prune module

    # For now, just return the original model
    print(f"Warning: Pruning not fully implemented - returning original model")
    return model