import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

# from .optimized_model import OptimizedUNet


class LightweightUNet(nn.Module):
    """
    Lightweight U-Net architecture with GroupNorm for image de-glaring.
    Designed for grayscale images.
    """
    def __init__(self, in_channels=1, out_channels=1, num_groups=8, features_start=8):
        """
        Initialize the lightweight U-Net model.

        Args:
            in_channels (int): Number of input channels (1 for grayscale).
            out_channels (int): Number of output channels (1 for grayscale).
            num_groups (int): Number of groups for GroupNorm.
                               Ensure this divides the feature channels at each stage,
                               or adjust logic in _block for small channel counts.
                               A common value is 8 or 16 or 32.
            features_start (int): Number of features in the first encoder block.
                                  Subsequent blocks will double this.
        """
        super().__init__()
        self.num_groups = num_groups

        f = [features_start, features_start*2, features_start*4, features_start*8, features_start*16]

        # Encoder (downsampling)
        self.enc1 = self._block(in_channels, f[0])
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.enc2 = self._block(f[0], f[1])
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.enc3 = self._block(f[1], f[2])
        self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.enc4 = self._block(f[2], f[3])
        self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = self._block(f[3], f[4])

        # Decoder (upsampling)
        self.upconv4 = nn.ConvTranspose2d(f[4], f[3], kernel_size=2, stride=2)
        self.dec4 = self._block(f[3] * 2, f[3])  # Concatenation: f[3] (from upconv) + f[3] (from enc4)
        self.upconv3 = nn.ConvTranspose2d(f[3], f[2], kernel_size=2, stride=2)
        self.dec3 = self._block(f[2] * 2, f[2])
        self.upconv2 = nn.ConvTranspose2d(f[2], f[1], kernel_size=2, stride=2)
        self.dec2 = self._block(f[1] * 2, f[1])
        self.upconv1 = nn.ConvTranspose2d(f[1], f[0], kernel_size=2, stride=2)
        self.dec1 = self._block(f[0] * 2, f[0])

        # Output layer
        self.output_conv = nn.Conv2d(f[0], out_channels, kernel_size=1)

    def _block(self, in_channels, features):
        """
        Create a convolutional block for the U-Net with GroupNorm.

        Args:
            in_channels (int): Number of input channels.
            features (int): Number of output channels for this block.

        Returns:
            nn.Sequential: Sequential container of layers.
        """
        # A robust way to set num_groups for GN:
        current_num_groups = self.num_groups
        if features < self.num_groups : # e.g. features = 8, num_groups = 32
            current_num_groups = features # each channel its own group (like InstanceNorm if groups=features)
                                     # or 1 (like LayerNorm if groups=1)
                                     # A common choice is to ensure features % current_num_groups == 0
                                     # Let's find largest divisor for features that is <= self.num_groups
            for i in range(min(self.num_groups, features), 0, -1):
                if features % i == 0:
                    current_num_groups = i
                    break
        elif features % self.num_groups != 0:
             # Find largest divisor for features that is <= self.num_groups
            for i in range(self.num_groups, 0, -1):
                if features % i == 0:
                    current_num_groups = i
                    break

        # If features_start is small (e.g. 8) and num_groups is also small (e.g. 8),
        # current_num_groups will be 8, which is fine.
        # If features_start=8, num_groups=32, then current_num_groups will be 8.

        return nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=current_num_groups, num_channels=features),
            nn.SiLU(inplace=True),
            nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=current_num_groups, num_channels=features),
            nn.SiLU(inplace=True)
        )

    def forward(self, x):
        """
        Forward pass through the network.
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

        out = self.output_conv(dec1)

        return out

class AttentionGate(nn.Module):
    """
    Attention Gate module for U-Net

    Helps the model focus on relevant features during the skip connection
    """
    def __init__(self, F_g, F_l, F_int):
        """
        Initialize attention gate

        Args:
            F_g (int): Number of channels in gating signal (from upper layer)
            F_l (int): Number of channels in input feature map
            F_int (int): Number of channels in intermediate representations
        """
        super(AttentionGate, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        """
        Forward pass

        Args:
            g (torch.Tensor): Gating signal from the higher layer
            x (torch.Tensor): Skip connection input

        Returns:
            torch.Tensor: Attention gated input
        """
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


class ResidualBlock(nn.Module):
    """
    Residual block for enhanced U-Net

    Improves gradient flow during training
    """
    def __init__(self, in_channels, out_channels):
        """
        Initialize residual block

        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
        """
        super(ResidualBlock, self).__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        Forward pass

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output after residual connection
        """
        residual = self.shortcut(x)
        out = self.conv_block(x)
        out += residual
        out = self.relu(out)
        return out


class EnhancedUNet(nn.Module):
    """
    Enhanced U-Net architecture for image de-glaring

    Includes residual connections, attention gates, and increased feature dimensions
    """
    def __init__(self, in_channels=1, out_channels=1):
        """
        Initialize the enhanced U-Net model

        Args:
            in_channels (int): Number of input channels (1 for grayscale)
            out_channels (int): Number of output channels (1 for grayscale)
        """
        super(EnhancedUNet, self).__init__()

        # Initial feature dimensions (doubled from original)
        self.init_features = 16

        # Encoder (downsampling) with residual blocks
        self.enc1 = ResidualBlock(in_channels, self.init_features)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc2 = ResidualBlock(self.init_features, self.init_features*2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc3 = ResidualBlock(self.init_features*2, self.init_features*4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc4 = ResidualBlock(self.init_features*4, self.init_features*8)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc5 = ResidualBlock(self.init_features*8, self.init_features*16)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck with dilated convolutions for larger receptive field
        self.bottleneck = nn.Sequential(
            nn.Conv2d(self.init_features*16, self.init_features*32, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(self.init_features*32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Conv2d(self.init_features*32, self.init_features*32, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(self.init_features*32),
            nn.ReLU(inplace=True)
        )

        # Attention gates - matching the exact channel dimensions
        self.attention5 = AttentionGate(F_g=self.init_features*16, F_l=self.init_features*16, F_int=self.init_features*8)
        self.attention4 = AttentionGate(F_g=self.init_features*8, F_l=self.init_features*8, F_int=self.init_features*4)
        self.attention3 = AttentionGate(F_g=self.init_features*4, F_l=self.init_features*4, F_int=self.init_features*2)
        self.attention2 = AttentionGate(F_g=self.init_features*2, F_l=self.init_features*2, F_int=self.init_features)
        self.attention1 = AttentionGate(F_g=self.init_features, F_l=self.init_features, F_int=self.init_features//2)

        # Decoder (upsampling)
        self.upconv5 = nn.ConvTranspose2d(self.init_features*32, self.init_features*16, kernel_size=2, stride=2)
        self.dec5 = ResidualBlock(self.init_features*32, self.init_features*16)

        self.upconv4 = nn.ConvTranspose2d(self.init_features*16, self.init_features*8, kernel_size=2, stride=2)
        self.dec4 = ResidualBlock(self.init_features*16, self.init_features*8)

        self.upconv3 = nn.ConvTranspose2d(self.init_features*8, self.init_features*4, kernel_size=2, stride=2)
        self.dec3 = ResidualBlock(self.init_features*8, self.init_features*4)

        self.upconv2 = nn.ConvTranspose2d(self.init_features*4, self.init_features*2, kernel_size=2, stride=2)
        self.dec2 = ResidualBlock(self.init_features*4, self.init_features*2)

        self.upconv1 = nn.ConvTranspose2d(self.init_features*2, self.init_features, kernel_size=2, stride=2)
        self.dec1 = ResidualBlock(self.init_features*2, self.init_features)

        # Output layer
        self.output = nn.Sequential(
            nn.Conv2d(self.init_features, out_channels, kernel_size=1),
            nn.Sigmoid()  # Ensure output is in [0, 1] range
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
        enc5 = self.enc5(self.pool4(enc4))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool5(enc5))

        # Decoder with attention-gated skip connections
        dec5 = self.upconv5(bottleneck)
        enc5 = self.attention5(dec5, enc5)
        dec5 = torch.cat((dec5, enc5), dim=1)
        dec5 = self.dec5(dec5)

        dec4 = self.upconv4(dec5)
        enc4 = self.attention4(dec4, enc4)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.dec4(dec4)

        dec3 = self.upconv3(dec4)
        enc3 = self.attention3(dec3, enc3)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.dec3(dec3)

        dec2 = self.upconv2(dec3)
        enc2 = self.attention2(dec2, enc2)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.dec2(dec2)

        dec1 = self.upconv1(dec2)
        enc1 = self.attention1(dec1, enc1)
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

    try:
        # Implement proper static quantization
        import torch.quantization

        # Define quantization configuration
        model.qconfig = torch.quantization.get_default_qconfig('fbgemm')

        # Prepare model for quantization
        model_prepared = torch.quantization.prepare(model)

        # Calibrate with the dataloader if provided
        if calibration_dataloader is not None:
            with torch.no_grad():
                for inputs, _ in calibration_dataloader:
                    model_prepared(inputs)

        # Convert to quantized model
        quantized_model = torch.quantization.convert(model_prepared)

        print(f"Model successfully quantized. Original size: {get_model_size_mb(model):.2f} MB, "
              f"Quantized size: {get_model_size_mb(quantized_model):.2f} MB")

        return quantized_model

    except (ImportError, AttributeError) as e:
        print(f"Warning: Quantization failed with error: {e} - returning original model")
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
    try:
        import torch.nn.utils.prune as prune

        # Create a copy of the model to prune
        pruned_model = copy.deepcopy(model)

        # Get total parameters before pruning
        total_params_before = sum(p.numel() for p in pruned_model.parameters() if p.requires_grad)

        # Apply global pruning to all Conv2d and Linear layers
        for name, module in pruned_model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                # L1 unstructured pruning (remove lowest magnitude weights)
                prune.l1_unstructured(module, name='weight', amount=amount)

                # Make pruning permanent
                prune.remove(module, 'weight')

        # Count parameters after pruning
        total_params_after = sum(p.numel() for p in pruned_model.parameters() if p.requires_grad)

        # Convert pruned weights to zeros
        for param in pruned_model.parameters():
            if param.requires_grad:
                param.data = param.data.to(torch.float32)

        print(f"Model pruned: {amount*100:.1f}% of weights removed. "
              f"Parameters: {total_params_before:,} → {total_params_after:,}")

        return pruned_model

    except (ImportError, AttributeError) as e:
        print(f"Warning: Pruning failed with error: {e} - returning original model")
        return model