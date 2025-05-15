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
    """
    def __init__(self, in_channels: int = 1, pretrained: bool = True):
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
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass
        
        Args:
            x (torch.Tensor): Input tensor [B, C, H, W], where C should match in_channels
                             Expected input size is [B, 1, 224, 224]
        
        Returns:
            torch.Tensor: Feature tensor [B, 512, 7, 7]
        """
        return self.features(x)


def get_resnet34_encoder(in_channels: int = 1, pretrained: bool = True) -> ResNet34Backbone:
    """Create ResNet34 feature encoder
    
    Args:
        in_channels (int): Number of input channels, typically 1 for grayscale ultrasound images
        pretrained (bool): Whether to use pretrained weights
    
    Returns:
        ResNet34Backbone: ResNet34 feature extractor
    """
    return ResNet34Backbone(in_channels=in_channels, pretrained=pretrained)


if __name__ == "__main__":
    # Test ResNet34 feature extractor using processed data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set paths and load data
    processed_data_dir = os.path.join("data", "processed")
    train_dataset_path = os.path.join(processed_data_dir, "train_transitions_dataset.json")
    
    print(f"Loading data from {train_dataset_path}")
    with open(train_dataset_path, 'r') as f:
        train_data = json.load(f)
    
    print(f"Loaded {len(train_data)} samples from processed dataset")
    
    # Create model
    backbone = get_resnet34_encoder(in_channels=1, pretrained=True).to(device)
    print(f"Model structure:\n{backbone}")
    
    # Test with real ultrasound images
    batch_size = 4
    sample_indices = np.random.choice(len(train_data), batch_size, replace=False)
    
    # Load and preprocess ultrasound images
    print("\nLoading and processing sample images...")
    input_tensor_list = []
    sample_ids = []
    
    for idx in sample_indices:
        sample = train_data[idx]
        img_path = os.path.join(processed_data_dir, sample["ft1_image_path"])
        sample_ids.append(os.path.basename(img_path))
        
        # Load and convert to tensor
        image = Image.open(img_path).convert('L')  # Convert to grayscale
        image_tensor = torch.tensor(np.array(image), dtype=torch.float32) / 255.0  # Normalize to [0,1]
        image_tensor = image_tensor.unsqueeze(0)  # Add channel dimension [1, H, W]
        input_tensor_list.append(image_tensor)
    
    # Stack tensors into a batch
    batch_tensor = torch.stack(input_tensor_list).to(device)
    print(f"Input batch shape: {batch_tensor.shape}")
    
    # Visualize some input images
    plt.figure(figsize=(15, 5))
    for i in range(batch_size):
        plt.subplot(1, batch_size, i+1)
        plt.imshow(batch_tensor[i, 0].cpu().numpy(), cmap='gray')
        plt.title(f"{sample_ids[i]}")
        plt.axis('off')
    plt.tight_layout()
    plt.savefig("sample_input_images.png")
    print("Saved input images visualization to sample_input_images.png")
    
    # Forward pass
    print("\nRunning inference through ResNet34 backbone...")
    with torch.no_grad():
        output = backbone(batch_tensor)
    
    # Display output shape
    print(f"Output tensor shape: {output.shape}")
    
    # Verify output shape matches expected: [B, 512, 7, 7]
    expected_shape = (batch_size, 512, 7, 7)
    assert output.shape == expected_shape, f"Output shape {output.shape} does not match expected {expected_shape}"
    
    print("Test passed! Output shape matches expected.")
    
    # Visualize feature maps (first few channels for first image)
    num_features_to_show = 8
    plt.figure(figsize=(16, 8))
    for i in range(num_features_to_show):
        plt.subplot(2, 4, i+1)
        feature_map = output[0, i].cpu().numpy()
        plt.imshow(feature_map, cmap='viridis')
        plt.title(f"Feature {i}")
        plt.colorbar()
        plt.axis('off')
    plt.tight_layout()
    plt.savefig("sample_feature_maps.png")
    print(f"Saved feature map visualization to sample_feature_maps.png")
    
    # Print feature statistics
    print(f"\nOutput feature statistics:")
    print(f"  Min: {output.min().item():.4f}")
    print(f"  Max: {output.max().item():.4f}")
    print(f"  Mean: {output.mean().item():.4f}")
    print(f"  Std: {output.std().item():.4f}")
    
    print("\nResNet34 backbone test with real ultrasound images completed successfully!") 