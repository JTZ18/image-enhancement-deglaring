import torch
import torch.nn as nn
import torch.nn.functional as F


class OptimizedUNet(nn.Module):
    """
    Optimized U-Net architecture for image de-glaring
    
    Features:
    - Reduced depth (fewer layers)
    - Fewer channels in each layer
    - No 5th encoder/decoder block (vs enhanced model)
    - Simplified attention mechanism
    - Group normalization instead of batch normalization
    - EfficientNet-inspired squeeze-and-excitation blocks
    """
    def __init__(self, in_channels=1, out_channels=1):
        """
        Initialize the optimized U-Net model
        
        Args:
            in_channels (int): Number of input channels (1 for grayscale)
            out_channels (int): Number of output channels (1 for grayscale)
        """
        super().__init__()
        
        # Initial feature dimensions
        self.init_features = 16
        
        # Encoder (downsampling)
        self.enc1 = self._block(in_channels, self.init_features, groups=1)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)  # AvgPool is faster than MaxPool
        
        self.enc2 = self._block(self.init_features, self.init_features*2)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        self.enc3 = self._block(self.init_features*2, self.init_features*4)
        self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        self.enc4 = self._block(self.init_features*4, self.init_features*8)
        self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(self.init_features*8, self.init_features*16, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, self.init_features*16),
            nn.SiLU(inplace=True),  # SiLU (Swish) is more efficient than ReLU
            nn.Conv2d(self.init_features*16, self.init_features*16, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, self.init_features*16),
            nn.SiLU(inplace=True)
        )
        
        # Attention blocks - simplified
        self.attention4 = ChannelAttention(self.init_features*8)
        self.attention3 = ChannelAttention(self.init_features*4)
        self.attention2 = ChannelAttention(self.init_features*2)
        self.attention1 = ChannelAttention(self.init_features)
        
        # Decoder (upsampling) - Simplified to use nearest interpolation
        self.upconv4 = self._upblock(self.init_features*16, self.init_features*8)
        self.dec4 = self._block(self.init_features*16, self.init_features*8)
        
        self.upconv3 = self._upblock(self.init_features*8, self.init_features*4)
        self.dec3 = self._block(self.init_features*8, self.init_features*4)
        
        self.upconv2 = self._upblock(self.init_features*4, self.init_features*2)
        self.dec2 = self._block(self.init_features*4, self.init_features*2)
        
        self.upconv1 = self._upblock(self.init_features*2, self.init_features)
        self.dec1 = self._block(self.init_features*2, self.init_features)
        
        # Output layer
        self.output = nn.Conv2d(self.init_features, out_channels, kernel_size=1)
    
    def _block(self, in_channels, features, groups=4):
        """
        Create an optimized convolutional block
        
        Args:
            in_channels (int): Number of input channels
            features (int): Number of output channels
            groups (int): Number of groups for GroupNorm
        
        Returns:
            nn.Sequential: Sequential container of layers
        """
        # Ensure groups isn't larger than features
        groups = min(groups, features)
        
        return nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(max(1, groups), features),  # GroupNorm is more efficient than BatchNorm
            nn.SiLU(inplace=True),  # SiLU (Swish) is more efficient than ReLU
            nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(max(1, groups), features),
            nn.SiLU(inplace=True)
        )
    
    def _upblock(self, in_channels, out_channels):
        """
        Create an upsampling block using interpolation (faster than ConvTranspose2d)
        
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            
        Returns:
            nn.Sequential: Sequential container for upsampling
        """
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(4, out_channels),
            nn.SiLU(inplace=True)
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
        
        # Decoder with attention-enhanced skip connections
        dec4 = self.upconv4(bottleneck)
        enc4 = self.attention4(enc4)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.dec4(dec4)
        
        dec3 = self.upconv3(dec4)
        enc3 = self.attention3(enc3)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.upconv2(dec3)
        enc2 = self.attention2(enc2)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.upconv1(dec2)
        enc1 = self.attention1(enc1)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.dec1(dec1)
        
        return self.output(dec1)


class ChannelAttention(nn.Module):
    """
    Lightweight channel attention mechanism inspired by squeeze-and-excitation blocks
    Helps the model focus on relevant features with minimal computational overhead
    """
    def __init__(self, channels, reduction=16):
        """
        Initialize channel attention module
        
        Args:
            channels (int): Number of input channels
            reduction (int): Reduction ratio for the bottleneck
        """
        super(ChannelAttention, self).__init__()
        reduced_channels = max(channels // reduction, 8)
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced_channels, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(reduced_channels, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x (torch.Tensor): Input tensor [B, C, H, W]
            
        Returns:
            torch.Tensor: Attention-weighted input
        """
        b, c, _, _ = x.size()
        avg = self.avg_pool(x).view(b, c)
        
        # Channel attention weights
        weights = self.fc(avg).view(b, c, 1, 1)
        
        # Apply attention
        return x * weights