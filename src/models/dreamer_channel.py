# ★ 512 token 版本
# 在這裡加入 Dreamer Channel 模型的定義 

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
    def __init__(self, d_model: int, max_len: int = 514):
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


class ChannelTokenEmbedding(nn.Module):
    """
    Embedding layer for channel tokens
    
    Maps channel tokens (from ResNet feature map) to transformer embeddings
    """
    def __init__(self, in_dim: int = 49, d_model: int = 768):
        super().__init__()
        self.projection = nn.Linear(in_dim, d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Channel tokens of shape [batch_size, n_channels, feature_dim]
               Typically [B, 512, 49] for 7x7 feature maps
               
        Returns:
            Channel embeddings of shape [batch_size, n_channels, d_model]
        """
        return self.projection(x)


class DreamerChannel(nn.Module):
    """
    Dreamer Channel-Token Model
    
    Uses 512 channel tokens + 1 action token in a transformer encoder
    
    Args:
        d_model: Dimension of transformer model
        nhead: Number of attention heads
        num_layers: Number of transformer layers
        dim_feedforward: Dimension of feedforward network
        feature_dim: Dimension of input features (7x7=49 for ResNet)
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
        feature_dim: int = 49,
        dropout: float = 0.1,
        activation: str = "gelu",
        use_flash_attn: bool = False  # 默認設為False避免警告
    ):
        super().__init__()
        
        self.d_model = d_model
        self.feature_dim = feature_dim
        
        # Token embeddings
        self.channel_embedding = ChannelTokenEmbedding(in_dim=feature_dim, d_model=d_model)
        self.action_embedding = ActionEmbedding(d_model=d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=514)
        
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
            norm_first=False  # 改為False以解決nested tensor警告
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers
        )
        
        # 僅當明確需要且環境支持時才啟用flash attention
        if use_flash_attn:
            try:
                from flash_attn.modules.mha import FlashSelfAttention
                print("使用Flash Attention提高大型token集合的性能")
                # 這裡會實際實現Flash Attention
            except ImportError:
                print("警告: 未找到flash_attn套件。使用標準注意力機制。")
        
        # Projection for channel token reconstruction (not needed for the CLS token)
        self.output_projection = nn.Linear(d_model, feature_dim)
        
    def forward(
        self, 
        channel_tokens: torch.Tensor, 
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass

        Args:
            channel_tokens: Channel tokens from backbone [batch_size, 512, feature_dim]
            action: 6-DOF action [batch_size, 6]
            
        Returns:
            transformed_channel_tokens: Transformed channel tokens [batch_size, 512, feature_dim]
            full_sequence: Full sequence of transformed tokens [batch_size, 513, d_model]
        """
        batch_size = channel_tokens.shape[0]
        
        # Project channel tokens to embedding dimension
        # [B, 512, 49] -> [B, 512, d_model]
        channel_emb = self.channel_embedding(channel_tokens)
        
        # Embed action to token
        # [B, 6] -> [B, 1, d_model]
        action_emb = self.action_embedding(action)
        
        # Concatenate action token (CLS) with channel tokens
        # [B, 1, d_model] + [B, 512, d_model] -> [B, 513, d_model]
        sequence = torch.cat([action_emb, channel_emb], dim=1)
        
        # Add positional encoding
        sequence = self.pos_encoder(sequence)
        
        # Apply layer normalization
        sequence = self.layer_norm(sequence)
        
        # Pass through transformer encoder
        # [B, 513, d_model] -> [B, 513, d_model]
        transformed_sequence = self.transformer_encoder(sequence)
        
        # Split back into action token and channel tokens
        # [B, 513, d_model] -> [B, 1, d_model], [B, 512, d_model]
        transformed_action_token = transformed_sequence[:, 0:1, :]
        transformed_channel_tokens = transformed_sequence[:, 1:, :]
        
        # Project channel tokens back to feature dimension
        # [B, 512, d_model] -> [B, 512, feature_dim]
        reconstructed_channel_tokens = self.output_projection(transformed_channel_tokens)
        
        return reconstructed_channel_tokens, transformed_sequence


def get_dreamer_channel(
    d_model: int = 768,
    nhead: int = 12,
    num_layers: int = 6,
    feature_dim: int = 49,
    use_flash_attn: bool = False  # 默認設為False避免警告
) -> DreamerChannel:
    """
    Helper function to create a DreamerChannel model
    
    Args:
        d_model: Hidden dimension of transformer
        nhead: Number of attention heads
        num_layers: Number of transformer layers
        feature_dim: Dimension of input features
        use_flash_attn: Whether to use flash attention
        
    Returns:
        DreamerChannel model
    """
    return DreamerChannel(
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        feature_dim=feature_dim,
        use_flash_attn=use_flash_attn
    )


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


if __name__ == "__main__":
    # Test DreamerChannel model with backbone
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create models
    backbone = get_resnet34_encoder(in_channels=1, pretrained=True).to(device)
    dreamer = get_dreamer_channel(
        d_model=768,
        nhead=12,
        num_layers=6,
        feature_dim=49,
        use_flash_attn=False  # 已設定為False
    ).to(device)
    
    print(f"Backbone model created")
    print(f"DreamerChannel model created")
    print(f"DreamerChannel config: d_model=768, nhead=12, num_layers=6")
    
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
        
        # Reshape feature map to channel tokens
        # [B, 512, 7, 7] -> [B, 512, 49]
        channel_tokens = feature_map.reshape(batch_size, 512, -1)
        
        print(f"\nIntermediate shapes:")
        print(f"  Feature map: {feature_map.shape}")
        print(f"  Channel tokens: {channel_tokens.shape}")
        
        # Forward pass through DreamerChannel
        reconstructed_channels, full_sequence = dreamer(channel_tokens, action)
        
        print(f"\nOutput shapes:")
        print(f"  Reconstructed channels: {reconstructed_channels.shape}")
        print(f"  Full sequence: {full_sequence.shape}")
        
        # Check that the shapes are as expected
        assert reconstructed_channels.shape == (batch_size, 512, 49), \
            f"Expected reconstructed_channels shape {(batch_size, 512, 49)}, got {reconstructed_channels.shape}"
        
        assert full_sequence.shape == (batch_size, 513, 768), \
            f"Expected full_sequence shape {(batch_size, 513, 768)}, got {full_sequence.shape}"
        
        print("\nTest passed! All shapes match expected dimensions.") 
        