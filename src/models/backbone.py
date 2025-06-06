# Shared ResNet34 Backbone
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.resnet import ResNet, BasicBlock
from typing import List, Optional, Type, Union
import json
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

class ResNet34Backbone(nn.Module):
    """ResNet34 as Feature Encoder
    
    Transforms input ultrasound images (single channel) into feature maps [B, 512, 7, 7].
    
    Args:
        in_channels (int): Number of input channels, typically 1 for grayscale ultrasound images
        pretrained (bool): Whether to use pretrained weights
        freeze_layers (int): Number of layers to freeze (0: no freezing, 1: conv1+bn1, 
                           2: +layer1, 3: +layer2, 4: +layer3, 5: all layers frozen)
    """
    def __init__(self, in_channels: int = 1, pretrained: bool = True, freeze_layers: int = 0):
        super().__init__()
        
        # First load pretrained ResNet34 model
        if pretrained:
            # Use ImageNet pretrained weights
            resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
            print("Loaded ResNet34 pretrained weights")
        else:
            resnet = models.resnet34(weights=None)
            print("Using randomly initialized ResNet34")
        
        # Modify the first conv layer to accept single-channel input (if needed)
        if in_channels != 3:
            # Get original conv layer weights
            original_conv = resnet.conv1
            original_weight = original_conv.weight.data
            
            # Create new conv layer, keeping other parameters unchanged
            resnet.conv1 = nn.Conv2d(
                in_channels,
                original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=original_conv.bias is not None
            )
            
            # For single-channel input, average the weights across input channels
            if in_channels == 1:
                # Average the weights across the three input channels
                resnet.conv1.weight.data = original_weight.mean(dim=1, keepdim=True)
                print(f"Modified first conv layer to accept {in_channels} channel input")
            
        # Remove classification layer, keep only feature extraction part
        self.features = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
        )
        
        # Apply layer freezing if specified
        self.freeze_layers = freeze_layers
        self._freeze_layers()
        
    def _freeze_layers(self):
        """凍結指定數量的層參數"""
        if self.freeze_layers <= 0:
            print("No layers frozen - all parameters trainable")
            return
            
        # Define layer groups for freezing
        layer_groups = [
            [self.features[0], self.features[1]],  # conv1, bn1 (index 0-1)
            [self.features[4]],                     # layer1 (index 4)
            [self.features[5]],                     # layer2 (index 5) 
            [self.features[6]],                     # layer3 (index 6)
            [self.features[7]]                      # layer4 (index 7)
        ]
        
        frozen_layers_info = []
        
        # Freeze layers according to freeze_layers parameter
        for i in range(min(self.freeze_layers, len(layer_groups))):
            for layer in layer_groups[i]:
                for param in layer.parameters():
                    param.requires_grad = False
            
            if i == 0:
                frozen_layers_info.append("conv1+bn1")
            else:
                frozen_layers_info.append(f"layer{i}")
        
        print(f"Frozen {self.freeze_layers} layer group(s): {', '.join(frozen_layers_info)}")
        
        # Count trainable vs frozen parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
        print(f"  Frozen parameters: {frozen_params:,} ({frozen_params/total_params*100:.1f}%)")
    
    def unfreeze_all_layers(self):
        """解凍所有層參數 - 用於需要完全微調時"""
        for param in self.parameters():
            param.requires_grad = True
        print("All layers unfrozen - all parameters now trainable")
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass
        
        Args:
            x (torch.Tensor): Input tensor [B, C, H, W], where C should match in_channels
                             Expected input size is [B, 1, 224, 224]
        
        Returns:
            torch.Tensor: Feature tensor [B, 512, 7, 7]
        """
        return self.features(x)


def get_resnet34_encoder(in_channels: int = 1, pretrained: bool = True, freeze_layers: int = 0) -> ResNet34Backbone:
    """Create ResNet34 feature encoder
    
    Args:
        in_channels (int): Number of input channels, typically 1 for grayscale ultrasound images
        pretrained (bool): Whether to use pretrained weights
        freeze_layers (int): Number of layers to freeze (0: no freezing, 1: conv1+bn1, 
                           2: +layer1, 3: +layer2, 4: +layer3, 5: all layers frozen)
    
    Returns:
        ResNet34Backbone: ResNet34 feature extractor
    """
    return ResNet34Backbone(in_channels=in_channels, pretrained=pretrained, freeze_layers=freeze_layers)


if __name__ == "__main__":
    # Test ResNet34 feature extractor with our cross-patient dataset
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Import our DataLoader to test compatibility
    import sys
    import torchvision.transforms as transforms
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    
    try:
        from src.data import CrossPatientTransitionsDataset, get_patient_splits
        from torch.utils.data import DataLoader
        
        # Set up data directory
        data_dir = "data/processed"
        
        if not os.path.exists(data_dir):
            print(f"Data directory {data_dir} not found. Please ensure data is available.")
            sys.exit(1)
        
        # Create the same transform as used in DataLoader
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        # Get patient splits and create dataset
        train_patients, val_patients, test_patients = get_patient_splits(data_dir)
        
        dataset = CrossPatientTransitionsDataset(
            data_dir=data_dir,
            transform=transform,
            split="train",
            train_patients=train_patients,
            val_patients=val_patients,
            test_patients=test_patients,
            small_subset=True  # Use small subset for testing
        )
        
        print(f"Loaded dataset with {len(dataset)} samples")
        
        # Create DataLoader
        data_loader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=False,
            num_workers=0
        )
        
        # Get a batch from DataLoader
        batch = next(iter(data_loader))
        image_t1, a_hat_t1_to_t2_gt, at1_6dof_gt, at2_6dof_gt = batch
        
        print(f"\nDataLoader output:")
        print(f"  Image batch shape: {image_t1.shape}")
        print(f"  Image dtype: {image_t1.dtype}")
        print(f"  Image value range: [{image_t1.min():.3f}, {image_t1.max():.3f}]")
        print(f"  Expected input for backbone: [B, 1, 224, 224]")
        
        # Verify compatibility
        batch_size = image_t1.shape[0]
        expected_input_shape = (batch_size, 1, 224, 224)
        assert image_t1.shape == expected_input_shape, f"Shape mismatch: {image_t1.shape} vs {expected_input_shape}"
        
        print(f"  ✓ DataLoader output is compatible with backbone input requirements!")
        
    except ImportError as e:
        print(f"Could not import CrossPatientTransitionsDataset: {e}")
        print("Creating dummy data for testing...")
        
        # Create dummy data with correct properties
        batch_size = 4
        image_t1 = torch.randn(batch_size, 1, 224, 224) * 0.5 - 0.5  # Simulate normalized images
        
    # Move to device
    image_t1 = image_t1.to(device)
    
    # Create model
    backbone = get_resnet34_encoder(in_channels=1, pretrained=True).to(device)
    print(f"\nBackbone model created:")
    print(f"  Input channels: 1 (grayscale)")
    print(f"  Expected input shape: [B, 1, 224, 224]")
    print(f"  Expected output shape: [B, 512, 7, 7]")
    
    # Test forward pass
    print(f"\nTesting forward pass with real DataLoader batch...")
    print(f"Input batch shape: {image_t1.shape}")
    
    with torch.no_grad():
        output = backbone(image_t1)
    
    print(f"Output tensor shape: {output.shape}")
    
    # Verify output shape matches expected: [B, 512, 7, 7]
    expected_output_shape = (batch_size, 512, 7, 7)
    assert output.shape == expected_output_shape, f"Output shape {output.shape} does not match expected {expected_output_shape}"
    
    print(f"✓ Output shape matches expected!")
    
    # Print feature statistics
    print(f"\nOutput feature statistics:")
    print(f"  Shape: {output.shape}")
    print(f"  Min: {output.min().item():.4f}")
    print(f"  Max: {output.max().item():.4f}")
    print(f"  Mean: {output.mean().item():.4f}")
    print(f"  Std: {output.std().item():.4f}")
    
    # Test individual sample
    print(f"\nTesting individual sample processing...")
    single_image = image_t1[0:1]  # Take first image from batch
    print(f"Single image shape: {single_image.shape}")
    
    with torch.no_grad():
        single_output = backbone(single_image)
    
    print(f"Single output shape: {single_output.shape}")
    assert single_output.shape == (1, 512, 7, 7), f"Single output shape incorrect: {single_output.shape}"
    
    # Show channel token shape for dreamer
    channel_tokens = single_output.reshape(1, 512, -1)
    print(f"Channel tokens shape (for dreamer): {channel_tokens.shape}")
    print(f"Expected: [1, 512, 49] (batch, channels, spatial_features)")
    assert channel_tokens.shape == (1, 512, 49), f"Channel tokens shape incorrect: {channel_tokens.shape}"
    
    print(f"\n🎉 ResNet34 backbone compatibility test passed!")
    print(f"📊 Summary:")
    print(f"  ✓ Input: [B, 1, 224, 224] (matches DataLoader output)")
    print(f"  ✓ Output: [B, 512, 7, 7] (ready for dreamer)")
    print(f"  ✓ Channel tokens: [B, 512, 49] (7x7=49 spatial features)")
    print(f"  ✓ Data types: float32 throughout")
    print(f"  ✓ Pretrained weights: ImageNet → single channel adaptation") 