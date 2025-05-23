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
from src.utils.transformation_utils import dof6_to_matrix, matrix_to_dof6


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
        T_composed = torch.matmul(T_a_hat_t1_to_t2_gt, T_a_prime_t2_hat)
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
        predicted_action_composed: torch.Tensor,  # a_t1_prime_composed (主要任務輸出)
        target_action_composed_gt: torch.Tensor,    # a_t1_prime_gt (主要任務目標)
        reconstructed_channels_t1: torch.Tensor,
        original_channels_t1: torch.Tensor,
        a_prime_t2_hat: torch.Tensor,             # Predicted action at t2 (輔助任務輸出)
        target_a_t2_prime_gt: torch.Tensor        # a_t2_prime_gt (輔助任務目標)
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Computes losses with configurable weights for main and auxiliary tasks.
        
        Main task: Predict final composed action at1 (corresponds to figure's at1 vs ât1)
        Auxiliary tasks: t2 action prediction and latent reconstruction (for training stability)
        """
        # 主要任務損失 - L_SmoothL1(at1, ât1) 對應圖片中的公式
        main_task_loss = self.smooth_l1_loss(predicted_action_composed, target_action_composed_gt)
        
        # 輔助任務損失 - 幫助訓練過程
        aux_t2_action_loss = self.smooth_l1_loss(a_prime_t2_hat, target_a_t2_prime_gt)
        aux_latent_loss = F.mse_loss(reconstructed_channels_t1, original_channels_t1)
        
        # 根據配置組合損失
        if self.hparams.primary_task_only:
            # 只使用主要任務損失 - 對應圖片中的評估方式
            total_loss = main_task_loss
            combined_action_loss = main_task_loss
        else:
            # 完整訓練損失 - 包含輔助任務幫助訓練
            combined_action_loss = main_task_loss + self.lambda_t2_action * aux_t2_action_loss
            total_loss = combined_action_loss + self.lambda_latent * aux_latent_loss
        
        loss_dict = {
            "total_loss": total_loss,
            "main_task_loss": main_task_loss,  # 這是您關心的主要指標
            "combined_action_loss": combined_action_loss,
            "aux_t2_action_loss": aux_t2_action_loss,
            "aux_latent_loss": aux_latent_loss,
            # 保持向後兼容性
            "action_loss_t1_prime": main_task_loss,
            "action_loss_t2_prime": aux_t2_action_loss,
            "latent_loss": aux_latent_loss
        }
        return total_loss, loss_dict
    
    def _get_original_channels_t1(self, image_t1: torch.Tensor) -> torch.Tensor:
        "Helper to get original channel tokens for loss calculation, ensuring no grad."
        with torch.no_grad():
            feature_map_t1_original = self.backbone(image_t1)
            return feature_map_t1_original.reshape(image_t1.shape[0], 512, -1).detach()

    def training_step(self, batch: Tuple[torch.Tensor, ...], batch_idx: int) -> torch.Tensor:
        # Batch structure: (image_t1, a_t1_gt, a_hat_t1_to_t2_gt, a_t1_prime_gt, a_t2_prime_gt)
        image_t1, a_t1_gt, a_hat_t1_to_t2_gt, a_t1_prime_gt, a_t2_prime_gt = batch
        
        system_outputs = self(image_t1, a_hat_t1_to_t2_gt)
        
        predicted_composed_action = system_outputs["predicted_action_composed"]
        reconstructed_channels_t1 = system_outputs["reconstructed_channels_t1"]
        a_prime_t2_hat = system_outputs["a_prime_t2_hat"]
        
        original_channels_t1 = self._get_original_channels_t1(image_t1)
        
        total_loss, loss_dict = self.compute_losses(
            predicted_composed_action, 
            a_t1_prime_gt, 
            reconstructed_channels_t1,
            original_channels_t1,
            a_prime_t2_hat,
            a_t2_prime_gt 
        )
        
        self.log("train_total_loss", total_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_main_task_loss", loss_dict["main_task_loss"], prog_bar=True, on_step=True, on_epoch=True)  # 主要關注指標
        self.log("train_combined_action_loss", loss_dict["combined_action_loss"], on_step=True, on_epoch=True)
        self.log("train_action_loss_t1p", loss_dict["action_loss_t1_prime"], on_step=False, on_epoch=True)
        self.log("train_action_loss_t2p", loss_dict["action_loss_t2_prime"], on_step=False, on_epoch=True)
        self.log("train_latent_loss", loss_dict["latent_loss"], on_step=True, on_epoch=True)
        
        return total_loss
    
    def validation_step(self, batch: Tuple[torch.Tensor, ...], batch_idx: int) -> torch.Tensor:
        image_t1, a_t1_gt, a_hat_t1_to_t2_gt, a_t1_prime_gt, a_t2_prime_gt = batch
        system_outputs = self(image_t1, a_hat_t1_to_t2_gt)
        
        predicted_composed_action = system_outputs["predicted_action_composed"]
        reconstructed_channels_t1 = system_outputs["reconstructed_channels_t1"]
        a_prime_t2_hat = system_outputs["a_prime_t2_hat"]
        
        original_channels_t1 = self._get_original_channels_t1(image_t1)
            
        total_loss, loss_dict = self.compute_losses(
            predicted_composed_action, 
            a_t1_prime_gt,
            reconstructed_channels_t1,
            original_channels_t1,
            a_prime_t2_hat,
            a_t2_prime_gt
        )
        
        self.log("val_loss", total_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_main_task_loss", loss_dict["main_task_loss"], prog_bar=True, on_step=False, on_epoch=True)  # 主要關注指標
        self.log("val_combined_action_loss", loss_dict["combined_action_loss"], on_step=False, on_epoch=True)
        self.log("val_action_loss_t1p", loss_dict["action_loss_t1_prime"], on_step=False, on_epoch=True)
        self.log("val_action_loss_t2p", loss_dict["action_loss_t2_prime"], on_step=False, on_epoch=True)
        self.log("val_latent_loss", loss_dict["latent_loss"], on_step=False, on_epoch=True)
        
        return total_loss
    
    def test_step(self, batch: Tuple[torch.Tensor, ...], batch_idx: int) -> Dict[str, torch.Tensor]:
        image_t1, a_t1_gt, a_hat_t1_to_t2_gt, a_t1_prime_gt, a_t2_prime_gt = batch
        system_outputs = self(image_t1, a_hat_t1_to_t2_gt)
        
        predicted_composed_action = system_outputs["predicted_action_composed"]
        reconstructed_channels_t1 = system_outputs["reconstructed_channels_t1"]
        a_prime_t2_hat = system_outputs["a_prime_t2_hat"]
        
        original_channels_t1 = self._get_original_channels_t1(image_t1)
            
        total_loss, loss_dict = self.compute_losses(
            predicted_composed_action, 
            a_t1_prime_gt,
            reconstructed_channels_t1,
            original_channels_t1,
            a_prime_t2_hat,
            a_t2_prime_gt
        )
        
        # For reporting, calculate MSE of the primary composed action prediction
        mse_composed_action = F.mse_loss(predicted_composed_action, a_t1_prime_gt)
        
        self.log("test_total_loss", total_loss, on_step=False, on_epoch=True)
        self.log("test_combined_action_loss", loss_dict["combined_action_loss"], on_step=False, on_epoch=True)
        self.log("test_action_loss_t1p", loss_dict["action_loss_t1_prime"], on_step=False, on_epoch=True)
        self.log("test_action_loss_t2p", loss_dict["action_loss_t2_prime"], on_step=False, on_epoch=True)
        self.log("test_latent_loss", loss_dict["latent_loss"], on_step=False, on_epoch=True)
        self.log("test_mse_composed_action", mse_composed_action, on_step=False, on_epoch=True)
        
        return {
            "test_total_loss": total_loss,
            "test_mse_composed_action": mse_composed_action,
            "predicted_action_composed": predicted_composed_action,
            "target_action_composed_gt": a_t1_prime_gt,
            "a_prime_t2_hat": a_prime_t2_hat, # Include for potential analysis
            "target_a_t2_prime_gt": a_t2_prime_gt # Include for potential analysis
        }
    
    def test_epoch_end(self, outputs: List[Dict[str, torch.Tensor]]):
        all_preds_composed = torch.cat([out["predicted_action_composed"] for out in outputs])
        all_targets_composed_gt = torch.cat([out["target_action_composed_gt"] for out in outputs])
        
        overall_mse_composed = F.mse_loss(all_preds_composed, all_targets_composed_gt)
        self.log("test_final_mse_composed", overall_mse_composed)
        
        per_dim_mse_composed = torch.mean((all_preds_composed - all_targets_composed_gt) ** 2, dim=0)
        dim_names = ["X", "Y", "Z", "Roll", "Pitch", "Yaw"]
        for i, dim_mse in enumerate(per_dim_mse_composed):
            self.log(f"test_mse_composed_{dim_names[i]}", dim_mse)

        # Optionally, calculate and log MSE for a_prime_t2_hat vs target_a_t2_prime_gt if needed
        # all_preds_a_t2_hat = torch.cat([out["a_prime_t2_hat"] for out in outputs])
        # all_targets_a_t2_gt = torch.cat([out["target_a_t2_prime_gt"] for out in outputs])
        # overall_mse_a_t2_hat = F.mse_loss(all_preds_a_t2_hat, all_targets_a_t2_gt)
        # self.log("test_final_mse_a_t2_hat", overall_mse_a_t2_hat)

def get_cardiac_dreamer_system(
    token_type: str = "channel",
    d_model: int = 768,
    nhead: int = 12,
    num_layers: int = 6,
    feature_dim: int = 49, 
    lr: float = 1e-4,
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
    a_t1_gt = torch.randn(batch_size, 6, device=device)
    a_hat_t1_to_t2_gt = torch.randn(batch_size, 6, device=device)
    a_t1_prime_gt = torch.randn(batch_size, 6, device=device) # Target for composed action
    a_t2_prime_gt = torch.randn(batch_size, 6, device=device) # Target for a_prime_t2_hat

    print(f"\nInput shapes for forward pass:")
    print(f"  Image (image_t1): {image_t1.shape}")
    print(f"  Action at t1 (a_t1_gt): {a_t1_gt.shape}")
    print(f"  Relative action t1->t2 (a_hat_t1_to_t2_gt): {a_hat_t1_to_t2_gt.shape}")
    print(f"  Target composed action (a_t1_prime_gt): {a_t1_prime_gt.shape}")
    print(f"  Target action at t2 (a_t2_prime_gt): {a_t2_prime_gt.shape}")
    
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
        a_t1_prime_gt, 
        reconstructed_channels_t1,
        original_channels_t1,
        a_prime_t2_hat,
        a_t2_prime_gt 
    )
    
    print(f"\nLoss values:")
    print(f"  Total loss: {total_loss.item():.4f}")
    print(f"  Combined Action SmoothL1Loss: {loss_dict['combined_action_loss'].item():.4f}")
    print(f"  Action SmoothL1Loss (t1_prime): {loss_dict['action_loss_t1_prime'].item():.4f}")
    print(f"  Action SmoothL1Loss (t2_prime): {loss_dict['action_loss_t2_prime'].item():.4f}")
    print(f"  Latent loss (t1 MSE): {loss_dict['latent_loss'].item():.4f}")
        
    print("\nTesting training_step...")
    system.train()
    optimizer = system.configure_optimizers()["optimizer"]
    optimizer.zero_grad()
    batch_data_train = (image_t1, a_t1_gt, a_hat_t1_to_t2_gt, a_t1_prime_gt, a_t2_prime_gt)
    train_loss = system.training_step(batch_data_train, 0)
    train_loss.backward()
    optimizer.step()
    print(f"  Training step ran, loss: {train_loss.item():.4f}")

    print("\nTesting validation_step...")
    system.eval()
    batch_data_val = (image_t1, a_t1_gt, a_hat_t1_to_t2_gt, a_t1_prime_gt, a_t2_prime_gt)
    val_loss = system.validation_step(batch_data_val, 0)
    print(f"  Validation step ran, loss: {val_loss.item():.4f}")

    print("\nTesting test_step...")
    system.eval()
    batch_data_test = (image_t1, a_t1_gt, a_hat_t1_to_t2_gt, a_t1_prime_gt, a_t2_prime_gt)
    test_outputs = system.test_step(batch_data_test, 0)
    print(f"  Test step ran, MSE (composed action): {test_outputs['test_mse_composed_action'].item():.4f}")

    print("\nTesting test_epoch_end...")
    system.eval()
    outputs_list = [system.test_step(batch_data_test, 0) for _ in range(3)]
    system.test_epoch_end(outputs_list)
    print(f"  Test epoch end ran (check logged metrics for test_final_mse_composed and per-dim MSEs)")

    print("\nBasic tests passed for CardiacDreamerSystem with SmoothL1Loss and new batch structure.")
    print("Ensure your DataLoader provides batches with the new structure:")
    print("  (image_t1, a_t1_gt, a_hat_t1_to_t2_gt, a_t1_prime_gt, a_t2_prime_gt)") 