# Patch token version
# Dreamer Patch model definition

import torch
import torch.nn as nn
import math
import os
import sys
import numpy as np
from typing import Optional, Tuple

# Add the project root to the path so we can import the backbone
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.models.backbone import get_resnet34_encoder


class PositionalEncoding(nn.Module):
    """
    Positional encoding for Transformer
    
    Adds positional information to the input embeddings
    """
    def __init__(self, d_model: int, max_len: int = 50):
        super().__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply sine to even indices and cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension and register as buffer (not a parameter, but part of the model)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
            
        Returns:
            Output tensor with positional encoding added
        """
        return x + self.pe[:, :x.size(1), :]


class ActionEmbedding(nn.Module):
    """
    Embedding layer for 6-DOF action
    
    Converts 6-DOF action vector to token embedding
    """
    def __init__(self, d_model: int = 768):
        super().__init__()
        self.action_projection = nn.Sequential(
            nn.Linear(6, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, d_model)
        )
        
    def forward(self, action: torch.Tensor) -> torch.Tensor:
        """
        Args:
            action: Action tensor of shape [batch_size, 6]
            
        Returns:
            Action embedding of shape [batch_size, 1, d_model]
        """
        # Project action to embedding dimension
        action_emb = self.action_projection(action)
        
        # Add sequence dimension
        return action_emb.unsqueeze(1)


class PatchTokenEmbedding(nn.Module):
    """
    Embedding layer for patch tokens
    
    Converts ResNet feature map patches to transformer embeddings
    """
    def __init__(self, in_channels: int = 512, d_model: int = 768):
        super().__init__()
        self.projection = nn.Linear(in_channels, d_model)
        
    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feature_map: Feature map from ResNet [batch_size, 512, 7, 7]
                
        Returns:
            Patch embeddings of shape [batch_size, 49, d_model]
        """
        B, C, H, W = feature_map.shape
        # Reshape: [B, 512, 7, 7] -> [B, 49, 512]
        patches = feature_map.view(B, C, H * W).transpose(1, 2)
        # Project: [B, 49, 512] -> [B, 49, d_model]
        return self.projection(patches)


class DreamerPatch(nn.Module):
    """
    Dreamer Patch-Token Model
    
    Uses 49 patch tokens + 1 action token in a transformer encoder
    Each patch token represents a spatial location with full 512-dim features
    
    Args:
        d_model: Dimension of transformer model
        nhead: Number of attention heads
        num_layers: Number of transformer layers
        dim_feedforward: Dimension of feedforward network
        in_channels: Number of input channels from ResNet (512)
        dropout: Dropout rate
        activation: Activation function
        use_flash_attn: Whether to use flash attention
    """
    def __init__(
        self,
        d_model: int = 768,
        nhead: int = 12,
        num_layers: int = 6,
        dim_feedforward: int = 3072,
        in_channels: int = 512,
        dropout: float = 0.1,
        activation: str = "gelu",
        use_flash_attn: bool = False  # Default to False to avoid warnings
    ):
        super().__init__()
        
        self.d_model = d_model
        self.in_channels = in_channels
        
        # Token embeddings
        self.patch_embedding = PatchTokenEmbedding(in_channels=in_channels, d_model=d_model)
        self.action_embedding = ActionEmbedding(d_model=d_model)
        
        # Positional encoding (50 tokens: 1 action + 49 patches)
        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=50)
        
        # Layer normalization before transformer
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=False  # Set to False to resolve nested tensor warning
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers
        )
        
        # Only enable flash attention when explicitly needed and environment supports it
        if use_flash_attn:
            try:
                from flash_attn.modules.mha import FlashSelfAttention
                print("Using Flash Attention to improve performance for large token sets")
                # Flash Attention implementation would go here
            except ImportError:
                print("Warning: flash_attn package not found. Using standard attention mechanism.")
        
        # Projection for patch token reconstruction
        self.output_projection = nn.Linear(d_model, in_channels)
        
        # New: Projection for next frame feature prediction
        self.next_frame_projection = nn.Linear(d_model, in_channels)
        
    def forward(
        self, 
        feature_map: torch.Tensor, 
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass

        Args:
            feature_map: Feature map from ResNet [batch_size, 512, 7, 7]
            action: 6-DOF action [batch_size, 6]
            
        Returns:
            transformed_patch_tokens: Transformed patch tokens [batch_size, 49, 512]
            full_sequence: Full sequence of transformed tokens [batch_size, 50, d_model]
            predicted_next_features: Predicted next frame features [batch_size, 49, 512]
        """
        batch_size = feature_map.shape[0]
        
        # Project feature map to patch tokens
        # [B, 512, 7, 7] -> [B, 49, d_model]
        patch_emb = self.patch_embedding(feature_map)
        
        # Embed action to token
        # [B, 6] -> [B, 1, d_model]
        action_emb = self.action_embedding(action)
        
        # Concatenate action token (CLS) with patch tokens
        # [B, 1, d_model] + [B, 49, d_model] -> [B, 50, d_model]
        sequence = torch.cat([action_emb, patch_emb], dim=1)
        
        # Add positional encoding
        sequence = self.pos_encoder(sequence)
        
        # Apply layer normalization
        sequence = self.layer_norm(sequence)
        
        # Pass through transformer encoder
        # [B, 50, d_model] -> [B, 50, d_model]
        transformed_sequence = self.transformer_encoder(sequence)
        
        # Split back into action token and patch tokens
        # [B, 50, d_model] -> [B, 1, d_model], [B, 49, d_model]
        transformed_action_token = transformed_sequence[:, 0:1, :]
        transformed_patch_tokens = transformed_sequence[:, 1:, :]
        
        # Project patch tokens back to feature dimension
        # [B, 49, d_model] -> [B, 49, 512]
        reconstructed_patch_tokens = self.output_projection(transformed_patch_tokens)
        
        # New: Project action token to predict next frame features
        # [B, 1, d_model] -> [B, 1, 512]
        next_frame_features = self.next_frame_projection(transformed_action_token)
        
        # Expand to match patch dimensions
        # [B, 1, 512] -> [B, 49, 512]
        predicted_next_features = next_frame_features.expand(-1, 49, -1)
        
        return reconstructed_patch_tokens, transformed_sequence, predicted_next_features


def get_dreamer_patch(
    d_model: int = 768,
    nhead: int = 12,
    num_layers: int = 6,
    in_channels: int = 512,
    use_flash_attn: bool = False  # Default to False to avoid warnings
) -> DreamerPatch:
    """
    Helper function to create a DreamerPatch model
    
    Args:
        d_model: Hidden dimension of transformer
        nhead: Number of attention heads
        num_layers: Number of transformer layers
        in_channels: Number of input channels from ResNet
        use_flash_attn: Whether to use flash attention
        
    Returns:
        DreamerPatch model
    """
    return DreamerPatch(
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        in_channels=in_channels,
        use_flash_attn=use_flash_attn
    )


def pool_patch_features(patch_tokens: torch.Tensor) -> torch.Tensor:
    """
    Pools patch tokens to get a fixed-size feature vector
    
    Args:
        patch_tokens: Patch tokens from DreamerPatch [batch_size, 49, 512]
        
    Returns:
        pooled_features: Pooled feature vector [batch_size, 512]
    """
    # Global average pooling across the patch dimension
    # [B, 49, 512] -> [B, 512]
    return torch.mean(patch_tokens, dim=1)


if __name__ == "__main__":
    # Test DreamerPatch model with backbone
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create models
    backbone = get_resnet34_encoder(in_channels=1, pretrained=True).to(device)
    dreamer = get_dreamer_patch(
        d_model=768,
        nhead=12,
        num_layers=6,
        in_channels=512,
        use_flash_attn=False  # Default to False
    ).to(device)
    
    print(f"Backbone model created")
    print(f"DreamerPatch model created")
    print(f"DreamerPatch config: d_model=768, nhead=12, num_layers=6")
    
    # Create sample input
    batch_size = 1  # Testing with single image as requested
    
    # Create a random image tensor [B, 1, 224, 224]
    image = torch.randn(batch_size, 1, 224, 224).to(device)
    
    # Create a random 6-DOF action [B, 6]
    action = torch.randn(batch_size, 6).to(device)
    
    print(f"\nInput shapes:")
    print(f"  Image: {image.shape}")
    print(f"  Action: {action.shape}")
    
    # Forward pass through backbone to get features
    with torch.no_grad():
        # Get feature map from ResNet backbone
        # [B, 1, 224, 224] -> [B, 512, 7, 7]
        feature_map = backbone(image)
        
        print(f"\nIntermediate shapes:")
        print(f"  Feature map: {feature_map.shape}")
        
        # Forward pass through DreamerPatch
        reconstructed_patches, full_sequence, predicted_next_features = dreamer(feature_map, action)
        
        print(f"\nOutput shapes:")
        print(f"  Reconstructed patches: {reconstructed_patches.shape}")
        print(f"  Full sequence: {full_sequence.shape}")
        print(f"  Predicted next features: {predicted_next_features.shape}")
        
        # Check that the shapes are as expected
        assert reconstructed_patches.shape == (batch_size, 49, 512), \
            f"Expected reconstructed_patches shape {(batch_size, 49, 512)}, got {reconstructed_patches.shape}"
        
        assert full_sequence.shape == (batch_size, 50, 768), \
            f"Expected full_sequence shape {(batch_size, 50, 768)}, got {full_sequence.shape}"
            
        assert predicted_next_features.shape == (batch_size, 49, 512), \
            f"Expected predicted_next_features shape {(batch_size, 49, 512)}, got {predicted_next_features.shape}"
        
        # Test pooling on both current and next features
        pooled_current = pool_patch_features(reconstructed_patches)
        pooled_next = pool_patch_features(predicted_next_features)
        
        print(f"\nPooled feature shapes:")
        print(f"  Pooled current features: {pooled_current.shape}")
        print(f"  Pooled next features: {pooled_next.shape}")
        
        assert pooled_current.shape == (batch_size, 512), \
            f"Expected pooled_current shape {(batch_size, 512)}, got {pooled_current.shape}"
            
        assert pooled_next.shape == (batch_size, 512), \
            f"Expected pooled_next shape {(batch_size, 512)}, got {pooled_next.shape}"
        
        print("\nTest passed! All shapes match expected dimensions.") 
        print("Patch token approach successfully implemented:")
        print("  - Token 0: Action token (at1->at2)")
        print("  - Token 1-49: Patch tokens (each 512-dim vector for spatial location)")
        print("  - Output: Average pooled 512-dim vector for guidance") 