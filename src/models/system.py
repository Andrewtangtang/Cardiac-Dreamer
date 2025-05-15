# LitModule   token_type 驅動不同 dreamer
# 在這裡加入 PyTorch Lightning System 模組的定義 

import os
import sys

# Add project root to path to ensure imports work regardless of where script is run from
current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_file_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Dict, Any, Tuple, Optional, List

from src.models.backbone import get_resnet34_encoder
from src.models.dreamer_channel import get_dreamer_channel, pool_features
from src.models.guidance import get_guidance_layer


class CardiacDreamerSystem(pl.LightningModule):
    """
    PyTorch Lightning module for the Cardiac Dreamer system
    
    Integrates all components: backbone, dreamer, and guidance layer.
    Handles training, validation, and testing logic.
    
    Args:
        token_type: Type of token strategy to use (default: "channel")
        d_model: Dimension of transformer model (default: 768)
        nhead: Number of attention heads (default: 12)
        num_layers: Number of transformer layers (default: 6)
        feature_dim: Dimension of spatial features (default: 49)
        in_channels: Number of input image channels (default: 1)
        use_pretrained: Whether to use pretrained backbone (default: True)
        lr: Learning rate (default: 1e-4)
        weight_decay: Weight decay for optimizer (default: 1e-5)
        lambda_latent: Weight for latent loss (default: 0.2)
        use_flash_attn: Whether to use flash attention (default: True)
    """
    def __init__(
        self,
        token_type: str = "channel",
        d_model: int = 768,
        nhead: int = 12,
        num_layers: int = 6,
        feature_dim: int = 49,
        in_channels: int = 1,
        use_pretrained: bool = True,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        lambda_latent: float = 0.2,
        use_flash_attn: bool = True
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Store hyperparameters
        self.token_type = token_type
        self.lambda_latent = lambda_latent
        self.lr = lr
        self.weight_decay = weight_decay
        
        # Create backbone
        self.backbone = get_resnet34_encoder(in_channels=in_channels, pretrained=use_pretrained)
        
        # Create dreamer based on token_type
        if token_type == "channel":
            self.dreamer = get_dreamer_channel(
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
                feature_dim=feature_dim,
                use_flash_attn=use_flash_attn
            )
        else:
            raise ValueError(f"Unsupported token_type: {token_type}")
        
        # Create guidance layer
        self.guidance = get_guidance_layer(
            feature_dim=512,  # Fixed at 512 for ResNet34
            hidden_dim=1024   # As specified in the architecture
        )
        
        # Initialize metrics
        self.train_mse = torch.nn.MSELoss()
        self.val_mse = torch.nn.MSELoss()
        self.test_mse = torch.nn.MSELoss()
        
    def forward(
        self, 
        image: torch.Tensor, 
        action: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the entire system
        
        Args:
            image: Input ultrasound image [batch_size, channels, height, width]
            action: Current action [batch_size, 6]
            
        Returns:
            Dictionary containing:
                - predicted_action: Predicted 6-DOF action
                - reconstructed_channels: Reconstructed channel tokens
                - full_sequence: Full sequence from transformer
        """
        batch_size = image.shape[0]
        
        # Extract features using backbone
        # [B, C, H, W] -> [B, 512, 7, 7]
        feature_map = self.backbone(image)
        
        # Reshape feature map to channel tokens
        # [B, 512, 7, 7] -> [B, 512, 49]
        channel_tokens = feature_map.reshape(batch_size, 512, -1)
        
        # Process through dreamer
        # [B, 512, 49] -> [B, 512, 49], [B, 513, d_model]
        reconstructed_channels, full_sequence = self.dreamer(channel_tokens, action)
        
        # Pool features
        # [B, 512, 49] -> [B, 512]
        pooled_features = pool_features(reconstructed_channels)
        
        # Generate action prediction
        # [B, 512] -> [B, 6]
        predicted_action = self.guidance(pooled_features)
        
        return {
            "predicted_action": predicted_action,
            "reconstructed_channels": reconstructed_channels,
            "full_sequence": full_sequence
        }
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler"""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        
        # Reduce LR on plateau
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1
            }
        }
    
    def compute_losses(
        self, 
        predicted_action: torch.Tensor, 
        target_action: torch.Tensor,
        reconstructed_channels: torch.Tensor,
        original_channels: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute total loss and individual loss components
        
        Args:
            predicted_action: Predicted 6-DOF action [batch_size, 6]
            target_action: Target 6-DOF action [batch_size, 6]
            reconstructed_channels: Reconstructed channel tokens [batch_size, 512, feature_dim]
            original_channels: Original channel tokens [batch_size, 512, feature_dim]
            
        Returns:
            total_loss: Combined loss value
            loss_dict: Dictionary with individual loss components
        """
        # Action prediction loss (MSE)
        action_loss = F.mse_loss(predicted_action, target_action)
        
        # Latent reconstruction loss
        latent_loss = F.mse_loss(reconstructed_channels, original_channels)
        
        # Total loss
        total_loss = action_loss + self.lambda_latent * latent_loss
        
        # Return loss components
        loss_dict = {
            "total_loss": total_loss,
            "action_loss": action_loss,
            "latent_loss": latent_loss
        }
        
        return total_loss, loss_dict
    
    def training_step(self, batch, batch_idx):
        """Training step"""
        image, action, target_action = batch
        batch_size = image.shape[0]
        
        # Forward pass
        feature_map = self.backbone(image)
        channel_tokens = feature_map.reshape(batch_size, 512, -1)
        reconstructed_channels, full_sequence = self.dreamer(channel_tokens, action)
        pooled_features = pool_features(reconstructed_channels)
        predicted_action = self.guidance(pooled_features)
        
        # Compute losses
        total_loss, loss_dict = self.compute_losses(
            predicted_action, 
            target_action,
            reconstructed_channels,
            channel_tokens.detach()  # Use original channel tokens as target
        )
        
        # Log metrics
        self.log("train_loss", total_loss, prog_bar=True)
        self.log("train_action_loss", loss_dict["action_loss"])
        self.log("train_latent_loss", loss_dict["latent_loss"])
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step"""
        image, action, target_action = batch
        batch_size = image.shape[0]
        
        # Forward pass
        feature_map = self.backbone(image)
        channel_tokens = feature_map.reshape(batch_size, 512, -1)
        reconstructed_channels, full_sequence = self.dreamer(channel_tokens, action)
        pooled_features = pool_features(reconstructed_channels)
        predicted_action = self.guidance(pooled_features)
        
        # Compute losses
        total_loss, loss_dict = self.compute_losses(
            predicted_action, 
            target_action,
            reconstructed_channels,
            channel_tokens.detach()
        )
        
        # Log metrics
        self.log("val_loss", total_loss, prog_bar=True)
        self.log("val_action_loss", loss_dict["action_loss"])
        self.log("val_latent_loss", loss_dict["latent_loss"])
        
        return total_loss
    
    def test_step(self, batch, batch_idx):
        """Test step"""
        image, action, target_action = batch
        batch_size = image.shape[0]
        
        # Forward pass
        feature_map = self.backbone(image)
        channel_tokens = feature_map.reshape(batch_size, 512, -1)
        reconstructed_channels, full_sequence = self.dreamer(channel_tokens, action)
        pooled_features = pool_features(reconstructed_channels)
        predicted_action = self.guidance(pooled_features)
        
        # Compute losses
        total_loss, loss_dict = self.compute_losses(
            predicted_action, 
            target_action,
            reconstructed_channels,
            channel_tokens.detach()
        )
        
        # Log metrics
        self.log("test_loss", total_loss)
        self.log("test_action_loss", loss_dict["action_loss"])
        self.log("test_latent_loss", loss_dict["latent_loss"])
        
        # Compute MSE for each degree of freedom
        for i, dof_name in enumerate(['X', 'Y', 'Z', 'Roll', 'Pitch', 'Yaw']):
            dof_mse = F.mse_loss(predicted_action[:, i], target_action[:, i])
            self.log(f"test_{dof_name}_mse", dof_mse)
        
        return total_loss


def get_cardiac_dreamer_system(
    token_type: str = "channel",
    d_model: int = 768,
    nhead: int = 12,
    num_layers: int = 6,
    feature_dim: int = 49,
    lr: float = 1e-4,
    lambda_latent: float = 0.2,
    use_flash_attn: bool = True
) -> CardiacDreamerSystem:
    """
    Helper function to create a CardiacDreamerSystem
    
    Args:
        token_type: Type of token strategy ("channel")
        d_model: Dimension of transformer model
        nhead: Number of attention heads
        num_layers: Number of transformer layers
        feature_dim: Dimension of spatial features
        lr: Learning rate
        lambda_latent: Weight for latent loss
        use_flash_attn: Whether to use flash attention
        
    Returns:
        CardiacDreamerSystem instance
    """
    return CardiacDreamerSystem(
        token_type=token_type,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        feature_dim=feature_dim,
        lr=lr,
        lambda_latent=lambda_latent,
        use_flash_attn=use_flash_attn
    )


if __name__ == "__main__":
    # Test the CardiacDreamerSystem
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create system
    system = get_cardiac_dreamer_system(
        token_type="channel",
        d_model=768,
        nhead=12,
        num_layers=6,
        feature_dim=49,
        lr=1e-4,
        lambda_latent=0.2,
        use_flash_attn=False  # Set to True if flash attention is installed
    ).to(device)
    
    print(f"CardiacDreamerSystem created with token_type='channel'")
    print(f"Configuration: d_model=768, nhead=12, num_layers=6, lambda_latent=0.2")
    
    # Create dummy batch
    batch_size = 2
    
    # Create random tensors for testing
    image = torch.randn(batch_size, 1, 224, 224).to(device)
    action = torch.randn(batch_size, 6).to(device)
    target_action = torch.randn(batch_size, 6).to(device)
    
    print(f"\nInput shapes:")
    print(f"  Image: {image.shape}")
    print(f"  Action: {action.shape}")
    print(f"  Target action: {target_action.shape}")
    
    # Forward pass through the entire system
    with torch.no_grad():
        system_outputs = system(image, action)
        
        predicted_action = system_outputs["predicted_action"]
        reconstructed_channels = system_outputs["reconstructed_channels"]
        full_sequence = system_outputs["full_sequence"]
        
        print(f"\nSystem output shapes:")
        print(f"  Predicted action: {predicted_action.shape}")
        print(f"  Reconstructed channels: {reconstructed_channels.shape}")
        print(f"  Full sequence: {full_sequence.shape}")
        
        # Compute losses
        feature_map = system.backbone(image)
        channel_tokens = feature_map.reshape(batch_size, 512, -1)
        
        total_loss, loss_dict = system.compute_losses(
            predicted_action,
            target_action,
            reconstructed_channels,
            channel_tokens
        )
        
        print(f"\nLoss values:")
        print(f"  Total loss: {total_loss.item():.4f}")
        print(f"  Action loss: {loss_dict['action_loss'].item():.4f}")
        print(f"  Latent loss: {loss_dict['latent_loss'].item():.4f}")
        
        # Verify optimizer and learning rate scheduler
        optimizer = system.configure_optimizers()
        print(f"\nOptimizer and LR scheduler configured")
        
        print("\nTest passed! CardiacDreamerSystem is ready for training.") 