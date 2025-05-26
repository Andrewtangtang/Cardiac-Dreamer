# LitModule   token_type drives different dreamer
# PyTorch Lightning System module definition

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
import matplotlib.pyplot as plt
import numpy as np

from src.models.backbone import get_resnet34_encoder
from src.models.dreamer_channel import get_dreamer_channel, pool_features
from src.models.guidance import get_guidance_layer
from src.utils.transformation_utils import dof6_to_matrix, matrix_to_dof6, matrix_inverse


class CardiacDreamerSystem(pl.LightningModule):
    """
    PyTorch Lightning module for the Cardiac Dreamer system
    
    Integrates all components: backbone, dreamer, and guidance layer.
    Handles training, validation, and testing logic, including action composition.
    Uses SmoothL1Loss for action predictions.
    
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
        lambda_t2_action: Weight for auxiliary t2 action loss (default: 1.0)
        smooth_l1_beta: Beta parameter for SmoothL1Loss (default: 1.0)
        use_flash_attn: Whether to use flash attention (default: True)
        primary_task_only: If True, only use main task loss (at1 prediction) (default: False)
    """
    def __init__(
        self,
        token_type: str = "channel",
        d_model: int = 768,
        nhead: int = 12,
        num_layers: int = 6,
        feature_dim: int = 49, # Spatial dimension of channel tokens (e.g., 7*7=49)
        in_channels: int = 1,
        use_pretrained: bool = True,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        lambda_latent: float = 0.2,
        lambda_t2_action: float = 1.0,  # Weight for auxiliary t2 action loss
        smooth_l1_beta: float = 1.0, # Added beta for SmoothL1Loss
        use_flash_attn: bool = True,
        primary_task_only: bool = False  # If True, only use main task loss (at1 prediction)
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.token_type = token_type
        self.lambda_latent = lambda_latent
        self.lambda_t2_action = lambda_t2_action
        self.lr = lr
        self.weight_decay = weight_decay
        
        # Initialize test step outputs collection for PyTorch Lightning v2.0.0 compatibility
        self.test_step_outputs = []
        
        # Initialize validation step outputs collection for scatter plots
        self.validation_step_outputs = []
        
        self.backbone = get_resnet34_encoder(in_channels=in_channels, pretrained=use_pretrained)
        
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
        
        self.guidance = get_guidance_layer(
            feature_dim=512, 
            hidden_dim=1024
        )
        
        self.smooth_l1_loss = nn.SmoothL1Loss(beta=smooth_l1_beta)
        self.test_mse_metric = torch.nn.MSELoss() # For final MSE reporting in test_epoch_end

    def forward(
        self, 
        image_t1: torch.Tensor, 
        a_hat_t1_to_t2_gt: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the entire system with action composition.
        
        Args:
            image_t1: Input ultrasound image at time t1 [batch_size, channels, height, width]
            a_hat_t1_to_t2_gt: Ground truth relative action from t1 to t2 [batch_size, 6]. Provided by dataset. used to input in dreamer to predict f_hat_t2.
            
        Returns:
            Dictionary containing:
                - predicted_action_composed: Predicted composed action a_t1_prime_composed [batch_size, 6]
                - reconstructed_channels_t1: Reconstructed channel tokens from image_t1 [batch_size, 512, feature_dim]
                - a_prime_t2_hat: Predicted action at t2 (output of guidance on f_hat_t2) [batch_size, 6]
                - predicted_next_feature_tokens: Predicted feature tokens for t2 (f_hat_t2) [batch_size, 512, feature_dim]
        """
        batch_size = image_t1.shape[0]
        
        feature_map_t1 = self.backbone(image_t1)
        channel_tokens_t1 = feature_map_t1.reshape(batch_size, 512, -1)
        
        reconstructed_tokens_t1, _, predicted_next_feature_tokens_f_hat_t2 = self.dreamer(
            channel_tokens_t1, a_hat_t1_to_t2_gt
        )
        
        pooled_f_hat_t2 = pool_features(predicted_next_feature_tokens_f_hat_t2)
        a_prime_t2_hat = self.guidance(pooled_f_hat_t2)
        
        T_a_hat_t1_to_t2_gt = dof6_to_matrix(a_hat_t1_to_t2_gt)
        T_a_prime_t2_hat = dof6_to_matrix(a_prime_t2_hat)

        T_a_hat_t1_to_t2_gt_inv = matrix_inverse(T_a_hat_t1_to_t2_gt)
        T_composed = torch.matmul(T_a_prime_t2_hat, T_a_hat_t1_to_t2_gt_inv)
        a_t1_prime_composed = matrix_to_dof6(T_composed)
        
        return {
            "predicted_action_composed": a_t1_prime_composed,
            "reconstructed_channels_t1": reconstructed_tokens_t1,
            "a_prime_t2_hat": a_prime_t2_hat,
            "predicted_next_feature_tokens": predicted_next_feature_tokens_f_hat_t2
        }
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=50,
            eta_min=1e-6
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1
            }
        }
    
    def compute_losses(
        self, 
        predicted_action_composed: torch.Tensor,  # Model's predicted composed action
        at1_6dof_gt: torch.Tensor,               # Ground truth at1_6dof from JSON (main task target)
        reconstructed_channels_t1: torch.Tensor,
        original_channels_t1: torch.Tensor,
        a_prime_t2_hat: torch.Tensor,            # Model's predicted t2 action
        at2_6dof_gt: torch.Tensor                # Ground truth at2_6dof from JSON (auxiliary task target)
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Computes losses with configurable weights for main and auxiliary tasks.
        
        Main task: Predict at1_6dof from image and action change (predicted_action_composed vs at1_6dof_gt)
        Auxiliary tasks: t2 action prediction (a_prime_t2_hat vs at2_6dof_gt) and latent reconstruction
        """
        # Main task loss - L_SmoothL1(predicted_action_composed, at1_6dof_gt)
        main_task_loss = self.smooth_l1_loss(predicted_action_composed, at1_6dof_gt)
        
        # Auxiliary task losses - help with training process
        aux_t2_action_loss = self.smooth_l1_loss(a_prime_t2_hat, at2_6dof_gt)
        aux_latent_loss = F.mse_loss(reconstructed_channels_t1, original_channels_t1)
        
        # Combine losses based on configuration
        if self.hparams.primary_task_only:
            # Only use main task loss - corresponds to evaluation mode
            total_loss = main_task_loss
        else:
            # Full training loss - includes auxiliary tasks to help training
            total_loss = main_task_loss + self.lambda_t2_action * aux_t2_action_loss + self.lambda_latent * aux_latent_loss
        
        loss_dict = {
            "total_loss": total_loss,
            "main_task_loss": main_task_loss,  # This is the main metric we care about
            "aux_t2_action_loss": aux_t2_action_loss,
            "aux_latent_loss": aux_latent_loss
        }
        return total_loss, loss_dict
    
    def _get_original_channels_t1(self, image_t1: torch.Tensor) -> torch.Tensor:
        "Helper to get original channel tokens for loss calculation, ensuring no grad."
        with torch.no_grad():
            feature_map_t1_original = self.backbone(image_t1)
            return feature_map_t1_original.reshape(image_t1.shape[0], 512, -1).detach()

    def training_step(self, batch: Tuple[torch.Tensor, ...], batch_idx: int) -> torch.Tensor:
        # Batch structure: (image_t1, a_hat_t1_to_t2_gt, at1_6dof_gt, at2_6dof_gt)
        image_t1, a_hat_t1_to_t2_gt, at1_6dof_gt, at2_6dof_gt = batch
        
        system_outputs = self(image_t1, a_hat_t1_to_t2_gt)
        
        predicted_composed_action = system_outputs["predicted_action_composed"]
        reconstructed_channels_t1 = system_outputs["reconstructed_channels_t1"]
        a_prime_t2_hat = system_outputs["a_prime_t2_hat"]
        
        original_channels_t1 = self._get_original_channels_t1(image_t1)
        
        total_loss, loss_dict = self.compute_losses(
            predicted_composed_action, 
            at1_6dof_gt,  # Main task: predict at1_6dof
            reconstructed_channels_t1,
            original_channels_t1,
            a_prime_t2_hat,
            at2_6dof_gt   # Auxiliary task: predict at2_6dof
        )
        
        # Simplified logging - only essential metrics
        self.log("train_total_loss", total_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_main_task_loss", loss_dict["main_task_loss"], prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_aux_t2_loss", loss_dict["aux_t2_action_loss"], on_step=False, on_epoch=True)
        self.log("train_aux_latent_loss", loss_dict["aux_latent_loss"], on_step=False, on_epoch=True)
        
        return total_loss
    
    def validation_step(self, batch: Tuple[torch.Tensor, ...], batch_idx: int) -> torch.Tensor:
        image_t1, a_hat_t1_to_t2_gt, at1_6dof_gt, at2_6dof_gt = batch
        system_outputs = self(image_t1, a_hat_t1_to_t2_gt)
        
        predicted_composed_action = system_outputs["predicted_action_composed"]
        reconstructed_channels_t1 = system_outputs["reconstructed_channels_t1"]
        a_prime_t2_hat = system_outputs["a_prime_t2_hat"]
        
        original_channels_t1 = self._get_original_channels_t1(image_t1)
            
        total_loss, loss_dict = self.compute_losses(
            predicted_composed_action, 
            at1_6dof_gt,  # Main task: predict at1_6dof
            reconstructed_channels_t1,
            original_channels_t1,
            a_prime_t2_hat,
            at2_6dof_gt   # Auxiliary task: predict at2_6dof
        )
        
        # Simplified logging - only essential metrics
        self.log("val_total_loss", total_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_main_task_loss", loss_dict["main_task_loss"], prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_aux_t2_loss", loss_dict["aux_t2_action_loss"], on_step=False, on_epoch=True)
        self.log("val_aux_latent_loss", loss_dict["aux_latent_loss"], on_step=False, on_epoch=True)
        
        # Collect predictions and ground truth for scatter plots
        self.validation_step_outputs.append({
            "predicted_action_composed": predicted_composed_action.detach().cpu(),
            "target_action_composed_gt": at1_6dof_gt.detach().cpu()
        })
        
        return total_loss
    
    def test_step(self, batch: Tuple[torch.Tensor, ...], batch_idx: int) -> Dict[str, torch.Tensor]:
        image_t1, a_hat_t1_to_t2_gt, at1_6dof_gt, at2_6dof_gt = batch
        system_outputs = self(image_t1, a_hat_t1_to_t2_gt)
        
        predicted_composed_action = system_outputs["predicted_action_composed"]
        reconstructed_channels_t1 = system_outputs["reconstructed_channels_t1"]
        a_prime_t2_hat = system_outputs["a_prime_t2_hat"]
        
        original_channels_t1 = self._get_original_channels_t1(image_t1)
            
        total_loss, loss_dict = self.compute_losses(
            predicted_composed_action, 
            at1_6dof_gt,  # Main task: predict at1_6dof
            reconstructed_channels_t1,
            original_channels_t1,
            a_prime_t2_hat,
            at2_6dof_gt   # Auxiliary task: predict at2_6dof
        )
        
        # For reporting, calculate MSE of the primary composed action prediction
        mse_main_task = F.mse_loss(predicted_composed_action, at1_6dof_gt)
        
        # Simplified logging - only essential metrics
        self.log("test_total_loss", total_loss, on_step=False, on_epoch=True)
        self.log("test_main_task_loss", loss_dict["main_task_loss"], on_step=False, on_epoch=True)
        self.log("test_main_task_mse", mse_main_task, on_step=False, on_epoch=True)
        self.log("test_aux_t2_loss", loss_dict["aux_t2_action_loss"], on_step=False, on_epoch=True)
        self.log("test_aux_latent_loss", loss_dict["aux_latent_loss"], on_step=False, on_epoch=True)
        
        self.test_step_outputs.append({
            "test_total_loss": total_loss,
            "test_main_task_mse": mse_main_task,
            "predicted_action_composed": predicted_composed_action,
            "target_action_composed_gt": at1_6dof_gt,
            "a_prime_t2_hat": a_prime_t2_hat,
            "target_a_t2_prime_gt": at2_6dof_gt
        })
        
        return {
            "test_total_loss": total_loss,
            "test_main_task_mse": mse_main_task,
            "predicted_action_composed": predicted_composed_action,
            "target_action_composed_gt": at1_6dof_gt,
            "a_prime_t2_hat": a_prime_t2_hat,
            "target_a_t2_prime_gt": at2_6dof_gt
        }
    
    def on_test_epoch_end(self):
        """PyTorch Lightning v2.0.0 compatible test epoch end hook"""
        if not self.test_step_outputs:
            return
            
        all_preds_composed = torch.cat([out["predicted_action_composed"] for out in self.test_step_outputs])
        all_targets_composed_gt = torch.cat([out["target_action_composed_gt"] for out in self.test_step_outputs])
        
        overall_mse_main_task = F.mse_loss(all_preds_composed, all_targets_composed_gt)
        self.log("test_final_main_task_mse", overall_mse_main_task)
        
        per_dim_mse_main_task = torch.mean((all_preds_composed - all_targets_composed_gt) ** 2, dim=0)
        dim_names = ["X", "Y", "Z", "Roll", "Pitch", "Yaw"]
        for i, dim_mse in enumerate(per_dim_mse_main_task):
            self.log(f"test_main_task_mse_{dim_names[i]}", dim_mse)

        # Clear the outputs for next test epoch
        self.test_step_outputs.clear()

    def on_validation_epoch_end(self):
        """Generate scatter plots for validation predictions vs ground truth"""
        if not self.validation_step_outputs:
            return
        
        # Check if trainer is available (not available during standalone testing)
        try:
            trainer = self.trainer
            if trainer is None:
                raise RuntimeError("Trainer is None")
        except RuntimeError:
            print("⚠️  Trainer not available - skipping validation scatter plots generation")
            self.validation_step_outputs.clear()
            return
        
        # Concatenate all predictions and ground truth
        all_preds = torch.cat([out["predicted_action_composed"] for out in self.validation_step_outputs])
        all_targets = torch.cat([out["target_action_composed_gt"] for out in self.validation_step_outputs])
        
        # Convert to numpy for plotting
        preds_np = all_preds.numpy()
        targets_np = all_targets.numpy()
        
        # Create scatter plots for each 6DOF dimension
        dim_names = ["X", "Y", "Z", "Roll", "Pitch", "Yaw"]
        
        # Create output directory for plots
        if hasattr(trainer, 'logger') and hasattr(trainer.logger, 'log_dir'):
            plots_dir = os.path.join(trainer.logger.log_dir, "validation_scatter_plots")
        else:
            plots_dir = "validation_scatter_plots"
        
        os.makedirs(plots_dir, exist_ok=True)
        
        # Generate individual scatter plots for each dimension
        for i, dim_name in enumerate(dim_names):
            plt.figure(figsize=(8, 8))
            
            # Extract data for this dimension
            pred_dim = preds_np[:, i]
            target_dim = targets_np[:, i]
            
            # Create scatter plot
            plt.scatter(target_dim, pred_dim, alpha=0.6, s=20, color='blue', edgecolors='black', linewidth=0.5)
            
            # Add perfect prediction line (y=x)
            min_val = min(target_dim.min(), pred_dim.min())
            max_val = max(target_dim.max(), pred_dim.max())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
            
            # Calculate correlation coefficient
            correlation = np.corrcoef(target_dim, pred_dim)[0, 1]
            
            # Calculate MSE for this dimension
            mse = np.mean((pred_dim - target_dim) ** 2)
            
            # Set labels and title
            plt.xlabel(f'Ground Truth {dim_name}', fontsize=12)
            plt.ylabel(f'Predicted {dim_name}', fontsize=12)
            plt.title(f'Validation: Predicted vs Ground Truth - {dim_name}\nCorr: {correlation:.3f}, MSE: {mse:.4f}', fontsize=14)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Make plot square
            plt.axis('equal')
            
            # Save plot
            current_epoch = getattr(self, 'current_epoch', 0)
            plot_path = os.path.join(plots_dir, f'validation_scatter_{dim_name.lower()}_epoch_{current_epoch:03d}.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
        
        # Create a combined plot with all 6 dimensions
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, dim_name in enumerate(dim_names):
            pred_dim = preds_np[:, i]
            target_dim = targets_np[:, i]
            
            axes[i].scatter(target_dim, pred_dim, alpha=0.6, s=15, color='blue', edgecolors='black', linewidth=0.3)
            
            # Add perfect prediction line
            min_val = min(target_dim.min(), pred_dim.min())
            max_val = max(target_dim.max(), pred_dim.max())
            axes[i].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
            
            # Calculate metrics
            correlation = np.corrcoef(target_dim, pred_dim)[0, 1]
            mse = np.mean((pred_dim - target_dim) ** 2)
            
            axes[i].set_xlabel(f'Ground Truth {dim_name}', fontsize=10)
            axes[i].set_ylabel(f'Predicted {dim_name}', fontsize=10)
            axes[i].set_title(f'{dim_name}\nCorr: {correlation:.3f}, MSE: {mse:.4f}', fontsize=11)
            axes[i].grid(True, alpha=0.3)
            axes[i].set_aspect('equal', adjustable='box')
        
        current_epoch = getattr(self, 'current_epoch', 0)
        plt.suptitle(f'Validation Predictions vs Ground Truth - Epoch {current_epoch}', fontsize=16)
        plt.tight_layout()
        
        # Save combined plot
        combined_plot_path = os.path.join(plots_dir, f'validation_scatter_combined_epoch_{current_epoch:03d}.png')
        plt.savefig(combined_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Validation scatter plots saved to: {plots_dir}")
        print(f"   - Individual plots: validation_scatter_[dimension]_epoch_{current_epoch:03d}.png")
        print(f"   - Combined plot: validation_scatter_combined_epoch_{current_epoch:03d}.png")
        
        # Clear outputs for next epoch
        self.validation_step_outputs.clear()

def get_cardiac_dreamer_system(
    token_type: str = "channel",
    d_model: int = 768,
    nhead: int = 12,
    num_layers: int = 6,
    feature_dim: int = 49, 
    lr: float = 1e-4,
    weight_decay: float = 1e-5,
    lambda_latent: float = 0.2,
    lambda_t2_action: float = 1.0,
    smooth_l1_beta: float = 1.0, # Added beta here
    use_flash_attn: bool = True,
    in_channels: int = 1, 
    use_pretrained: bool = True,
    primary_task_only: bool = False  # Added missing parameter
) -> CardiacDreamerSystem:
    return CardiacDreamerSystem(
        token_type=token_type,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        feature_dim=feature_dim,
        lr=lr,
        weight_decay=weight_decay,
        lambda_latent=lambda_latent,
        lambda_t2_action=lambda_t2_action,
        smooth_l1_beta=smooth_l1_beta, # Pass beta
        use_flash_attn=use_flash_attn,
        in_channels=in_channels, 
        use_pretrained=use_pretrained,
        primary_task_only=primary_task_only  # Pass the parameter
    )


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    TOKEN_TYPE = "channel"
    D_MODEL = 768
    NHEAD = 12
    NUM_LAYERS = 6
    FEATURE_DIM = 49
    LR = 1e-4
    LAMBDA_LATENT = 0.2
    LAMBDA_T2_ACTION = 1.0
    SMOOTH_L1_BETA = 1.0
    USE_FLASH_ATTN = False
    IN_CHANNELS = 1
    USE_PRETRAINED = True

    system = get_cardiac_dreamer_system(
        token_type=TOKEN_TYPE,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        feature_dim=FEATURE_DIM,
        lr=LR,
        lambda_latent=LAMBDA_LATENT,
        lambda_t2_action=LAMBDA_T2_ACTION,
        smooth_l1_beta=SMOOTH_L1_BETA,
        use_flash_attn=USE_FLASH_ATTN,
        in_channels=IN_CHANNELS,
        use_pretrained=USE_PRETRAINED
    ).to(device)
    
    print(f"CardiacDreamerSystem created with SmoothL1Loss (beta={SMOOTH_L1_BETA})")
    
    batch_size = 2
    image_t1 = torch.randn(batch_size, IN_CHANNELS, 224, 224, device=device)
    a_hat_t1_to_t2_gt = torch.randn(batch_size, 6, device=device)
    at1_6dof_gt = torch.randn(batch_size, 6, device=device) # Target: JSON at1_6dof (main task)
    at2_6dof_gt = torch.randn(batch_size, 6, device=device) # Target: JSON at2_6dof (auxiliary task)

    print(f"\nInput shapes for forward pass:")
    print(f"  Image (image_t1): {image_t1.shape}")
    print(f"  Relative action t1->t2 (a_hat_t1_to_t2_gt): {a_hat_t1_to_t2_gt.shape}")
    print(f"  Target at1_6dof (at1_6dof_gt): {at1_6dof_gt.shape}")
    print(f"  Target at2_6dof (at2_6dof_gt): {at2_6dof_gt.shape}")
    
    system.eval()
    with torch.no_grad():
        system_outputs = system(image_t1, a_hat_t1_to_t2_gt)
        predicted_action_composed = system_outputs["predicted_action_composed"]
        reconstructed_channels_t1 = system_outputs["reconstructed_channels_t1"]
        a_prime_t2_hat = system_outputs["a_prime_t2_hat"]
        
        print(f"\nSystem output shapes:")
        print(f"  Predicted composed action (a_t1_prime_composed): {predicted_action_composed.shape}")
        print(f"  Reconstructed channels (t1): {reconstructed_channels_t1.shape}")
        print(f"  Predicted action at t2 (a_prime_t2_hat): {a_prime_t2_hat.shape}")

    original_channels_t1 = system._get_original_channels_t1(image_t1)
    total_loss, loss_dict = system.compute_losses(
        predicted_action_composed,
        at1_6dof_gt, 
        reconstructed_channels_t1,
        original_channels_t1,
        a_prime_t2_hat,
        at2_6dof_gt 
    )
    
    print(f"\nLoss values:")
    print(f"  Total loss: {total_loss.item():.4f}")
    print(f"  Main Task Loss: {loss_dict['main_task_loss'].item():.4f}")
    print(f"  Aux T2 Action Loss: {loss_dict['aux_t2_action_loss'].item():.4f}")
    print(f"  Aux Latent Loss: {loss_dict['aux_latent_loss'].item():.4f}")
        
    print("\nTesting training_step...")
    system.train()
    optimizer = system.configure_optimizers()["optimizer"]
    optimizer.zero_grad()
    batch_data_train = (image_t1, a_hat_t1_to_t2_gt, at1_6dof_gt, at2_6dof_gt)
    train_loss = system.training_step(batch_data_train, 0)
    train_loss.backward()
    optimizer.step()
    print(f"  Training step ran, loss: {train_loss.item():.4f}")

    print("\nTesting validation_step...")
    system.eval()
    batch_data_val = (image_t1, a_hat_t1_to_t2_gt, at1_6dof_gt, at2_6dof_gt)
    val_loss = system.validation_step(batch_data_val, 0)
    print(f"  Validation step ran, loss: {val_loss.item():.4f}")

    print("\nTesting test_step...")
    system.eval()
    batch_data_test = (image_t1, a_hat_t1_to_t2_gt, at1_6dof_gt, at2_6dof_gt)
    test_outputs = system.test_step(batch_data_test, 0)
    print(f"  Test step ran, MSE (main task): {test_outputs['test_main_task_mse'].item():.4f}")

    print("\nTesting test_epoch_end...")
    system.eval()
    system.on_test_epoch_end()
    print(f"  Test epoch end ran (check logged metrics for test_final_main_task_mse and per-dim MSEs)")

    print("\nTesting validation_epoch_end...")
    system.eval()
    system.on_validation_epoch_end()
    print(f"  Validation epoch end ran (check logged metrics for validation scatter plots)")

    print("\nBasic tests passed for CardiacDreamerSystem with simplified loss structure.")
    print("Ensure your DataLoader provides batches with the new structure:")
    print("  (image_t1, a_hat_t1_to_t2_gt, at1_6dof_gt, at2_6dof_gt)")
    print("Where:")
    print("  - image_t1: from JSON ft1_image_path")
    print("  - a_hat_t1_to_t2_gt: from JSON action_change_6dof")
    print("  - at1_6dof_gt: from JSON at1_6dof (main task target)")
    print("  - at2_6dof_gt: from JSON at2_6dof (auxiliary task target)") 