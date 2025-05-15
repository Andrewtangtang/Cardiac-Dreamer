# 共用 (512+q → 1024 → 512 → 6)
import torch
import torch.nn as nn
import os
import sys
import numpy as np
from typing import Optional, Tuple

# Add the project root to the path so we can import other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.models.backbone import get_resnet34_encoder
from src.models.dreamer_channel import get_dreamer_channel


class GuidanceLayer(nn.Module):
    """
    Guidance Layer for Cardiac Ultrasound Probe Navigation
    
    Takes feature vectors from DreamerChannel,
    outputs 6-DOF action prediction.
    
    Structure: MLP with 512 → 1024 → 512 → 6
    
    Args:
        feature_dim: Dimension of input feature vector (default: 512)
        hidden_dim: Dimension of hidden layer (default: 1024)
    """
    def __init__(
        self,
        feature_dim: int = 512,
        hidden_dim: int = 1024,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        
        # MLP for guidance: 512 → 1024 → 512 → 6
        self.mlp = nn.Sequential(
            # First layer: 512 → 1024
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # Second layer: 1024 → 512
            nn.Linear(hidden_dim, feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # Output layer: 512 → 6 (6-DOF action)
            nn.Linear(feature_dim, 6)
        )
        
    def forward(
        self,
        features: torch.Tensor,
        query: torch.Tensor = None  # Keep query for backward compatibility, but it's not used
    ) -> torch.Tensor:
        """
        Forward pass through the guidance layer
        
        Args:
            features: Feature tensor from DreamerChannel [batch_size, feature_dim]
            query: Unused parameter kept for backward compatibility
            
        Returns:
            actions: Predicted 6-DOF actions [batch_size, 6]
        """
        # Sanity check for dimensions
        batch_size = features.shape[0]
        assert features.shape == (batch_size, self.feature_dim), \
            f"Expected features shape [B, {self.feature_dim}], got {features.shape}"
        
        # Pass through MLP
        # [B, feature_dim] -> [B, 6]
        actions = self.mlp(features)
        
        return actions


def pool_features(channel_tokens: torch.Tensor) -> torch.Tensor:
    """
    Pools channel tokens to get a fixed-size feature vector
    
    Args:
        channel_tokens: Channel tokens from DreamerChannel [batch_size, 512, feature_dim]
        
    Returns:
        pooled_features: Pooled feature vector [batch_size, 512]
    """
    # Global average pooling across the spatial dimension
    # [B, 512, feature_dim] -> [B, 512]
    return torch.mean(channel_tokens, dim=2)


def get_guidance_layer(
    feature_dim: int = 512,
    query_dim: int = 0,  # Kept for backward compatibility
    hidden_dim: int = 1024
) -> GuidanceLayer:
    """
    Helper function to create a Guidance Layer
    
    Args:
        feature_dim: Dimension of input feature vector
        query_dim: Unused parameter kept for backward compatibility
        hidden_dim: Dimension of hidden layer
        
    Returns:
        GuidanceLayer: Guidance layer model
    """
    return GuidanceLayer(
        feature_dim=feature_dim,
        hidden_dim=hidden_dim
    )


if __name__ == "__main__":
    # Test Guidance Layer with backbone and dreamer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Parameters
    batch_size = 1
    
    # Create models
    backbone = get_resnet34_encoder(in_channels=1, pretrained=True).to(device)
    dreamer = get_dreamer_channel(
        d_model=768,
        nhead=12,
        num_layers=6,
        feature_dim=49,
        use_flash_attn=False
    ).to(device)
    guidance = get_guidance_layer(
        feature_dim=512,
        hidden_dim=1024
    ).to(device)
    
    print("Models created:")
    print(f"  - ResNet34 Backbone")
    print(f"  - DreamerChannel (d_model=768, nhead=12, num_layers=6)")
    print(f"  - Guidance Layer (feature_dim=512, hidden_dim=1024)")
    
    # Create sample inputs
    # Create a random image tensor [B, 1, 224, 224]
    image = torch.randn(batch_size, 1, 224, 224).to(device)
    
    # Create a random 6-DOF action [B, 6]
    action = torch.randn(batch_size, 6).to(device)
    
    print(f"\nInput shapes:")
    print(f"  Image: {image.shape}")
    print(f"  Action: {action.shape}")
    
    # Forward pass through the entire pipeline
    with torch.no_grad():
        # Get feature map from backbone
        # [B, 1, 224, 224] -> [B, 512, 7, 7]
        feature_map = backbone(image)
        print(f"\nBackbone output shape: {feature_map.shape}")
        
        # Reshape feature map to channel tokens
        # [B, 512, 7, 7] -> [B, 512, 49]
        channel_tokens = feature_map.reshape(batch_size, 512, -1)
        print(f"Channel tokens shape: {channel_tokens.shape}")
        
        # Forward pass through DreamerChannel
        # [B, 512, 49] -> [B, 512, 49]
        reconstructed_channels, full_sequence = dreamer(channel_tokens, action)
        print(f"DreamerChannel outputs:")
        print(f"  - Reconstructed channels: {reconstructed_channels.shape}")
        print(f"  - Full sequence: {full_sequence.shape}")
        
        # Pool features from reconstructed channels
        # [B, 512, 49] -> [B, 512]
        pooled_features = pool_features(reconstructed_channels)
        print(f"Pooled features shape: {pooled_features.shape}")
        
        # Forward pass through Guidance Layer
        # [B, 512] -> [B, 6]
        predicted_action = guidance(pooled_features)
        print(f"Predicted action shape: {predicted_action.shape}")
        
        # Verify test passed
        assert predicted_action.shape == (batch_size, 6), \
            f"Expected predicted_action shape [{batch_size}, 6], got {predicted_action.shape}"
        print("\nTest passed! Full pipeline is working correctly.") 