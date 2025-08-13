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
import gc  # 添加垃圾回收

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
        freeze_backbone_layers: Number of ResNet34 layers to freeze (default: 0)
                               0: no freezing, 1: conv1+bn1, 2: +layer1, 3: +layer2, 4: +layer3
        lr: Learning rate (default: 1e-4)
        weight_decay: Weight decay for optimizer (default: 1e-5)
        lambda_t2_action: Weight for auxiliary t2 action loss (default: 1.0)
        smooth_l1_beta: Beta parameter for SmoothL1Loss (default: 1.0)
        use_flash_attn: Whether to use flash attention (default: True)
        primary_task_only: If True, only use main task loss (at1 prediction) (default: False)
        scheduler_type: Type of scheduler to use (default: "cosine")
        scheduler_config: Configuration for the scheduler (default: None)
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
        freeze_backbone_layers: int = 0,  # New parameter for freezing ResNet34 layers
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        lambda_t2_action: float = 1.0,  # Weight for auxiliary t2 action loss
        smooth_l1_beta: float = 1.0, # Added beta for SmoothL1Loss
        use_flash_attn: bool = True,
        primary_task_only: bool = False,  # If True, only use main task loss (at1 prediction)
        scheduler_type: str = "cosine",   # NEW: scheduler type
        scheduler_config: dict = None     # NEW: scheduler configuration
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.token_type = token_type
        self.lambda_t2_action = lambda_t2_action
        self.lr = lr
        self.weight_decay = weight_decay
        self.scheduler_type = scheduler_type
        self.scheduler_config = scheduler_config or {}
        
        # Initialize test step outputs collection for PyTorch Lightning v2.0.0 compatibility
        self.test_step_outputs = []
        
        # Initialize validation step outputs collection for scatter plots
        self.validation_step_outputs = []
        
        # 添加記憶體管理參數
        self.max_validation_outputs = 1000  # 限制驗證輸出數量
        self.max_test_outputs = 1000        # 限制測試輸出數量
        
        # Initialize backbone with layer freezing support
        self.backbone = get_resnet34_encoder(
            in_channels=in_channels, 
            pretrained=use_pretrained,
            freeze_layers=freeze_backbone_layers  # Pass the freeze parameter
        )
        
        # Channel-token Dreamer only
        self.dreamer = get_dreamer_channel(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            feature_dim=feature_dim, 
            use_flash_attn=use_flash_attn
        )
        
        self.guidance = get_guidance_layer(
            feature_dim=512, 
            hidden_dim=1024
        )
        
        self.smooth_l1_loss = nn.SmoothL1Loss(beta=smooth_l1_beta)
        self.test_mse_metric = torch.nn.MSELoss() # For final MSE reporting in test_epoch_end
        
        # 打印模型結構信息
        self._print_model_info()
        
    def _print_model_info(self):
        """打印模型的參數統計信息"""
        print(f"\n=== CardiacDreamerSystem Model Info ===")
        print(f"Token type: {self.token_type}")
        print(f"Frozen backbone layers: {self.hparams.freeze_backbone_layers}")
        print(f"Scheduler type: {self.scheduler_type}")
        
        # 統計各組件的參數
        backbone_total = sum(p.numel() for p in self.backbone.parameters())
        backbone_trainable = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        
        dreamer_params = sum(p.numel() for p in self.dreamer.parameters())
        guidance_params = sum(p.numel() for p in self.guidance.parameters())
        
        total_params = sum(p.numel() for p in self.parameters())
        total_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"Backbone: {backbone_trainable:,}/{backbone_total:,} trainable ({backbone_trainable/backbone_total*100:.1f}%)")
        print(f"Dreamer: {dreamer_params:,} parameters")
        print(f"Guidance: {guidance_params:,} parameters")
        print(f"Total: {total_trainable:,}/{total_params:,} trainable ({total_trainable/total_params*100:.1f}%)")
        print("=" * 40)
        
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
                - a_prime_t2_hat: Predicted action at t2 (output of guidance on f_hat_t2) [batch_size, 6]
                - predicted_next_feature_tokens: Predicted feature tokens for t2 (f_hat_t2) [batch_size, 512, feature_dim]
        """
        batch_size = image_t1.shape[0]
        
        feature_map_t1 = self.backbone(image_t1)
        
        # Channel-token path
        channel_tokens_t1 = feature_map_t1.reshape(batch_size, 512, -1)
        
        predicted_next_feature_tokens_f_hat_t2, _ = self.dreamer(
            channel_tokens_t1, a_hat_t1_to_t2_gt
        )
        
        pooled_f_hat_t2 = pool_features(predicted_next_feature_tokens_f_hat_t2)
        
        a_prime_t2_hat = self.guidance(pooled_f_hat_t2)
        
        T_a_hat_t1_to_t2_gt = dof6_to_matrix(a_hat_t1_to_t2_gt)
        T_a_prime_t2_hat = dof6_to_matrix(a_prime_t2_hat)

        T_composed = torch.matmul(T_a_hat_t1_to_t2_gt,T_a_prime_t2_hat)
        a_t1_prime_composed = matrix_to_dof6(T_composed)
        
        return {
            "predicted_action_composed": a_t1_prime_composed,
            "a_prime_t2_hat": a_prime_t2_hat,
            "predicted_next_feature_tokens": pooled_f_hat_t2
        }
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        
        # Configure scheduler based on type
        if self.scheduler_type == "cosine":
            # Standard CosineAnnealingLR
            T_max = int(self.scheduler_config.get("T_max", 50))
            eta_min = float(self.scheduler_config.get("eta_min", 1e-6))
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=T_max,
                eta_min=eta_min
            )
            print(f"Using CosineAnnealingLR: T_max={T_max}, eta_min={eta_min}")
            
        elif self.scheduler_type == "cosine_warm_restarts":
            # CosineAnnealingWarmRestarts - for escaping local minima
            T_0 = int(self.scheduler_config.get("T_0", 50))
            T_mult = int(self.scheduler_config.get("T_mult", 1))
            eta_min = float(self.scheduler_config.get("eta_min", 1e-7))
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=T_0,
                T_mult=T_mult,
                eta_min=eta_min
            )
            print(f"Using CosineAnnealingWarmRestarts: T_0={T_0}, T_mult={T_mult}, eta_min={eta_min}")
            
        elif self.scheduler_type == "onecycle":
            # OneCycleLR - for fast convergence
            max_lr = float(self.scheduler_config.get("max_lr", self.lr * 3))
            total_steps = int(self.scheduler_config.get("total_steps", 100))
            pct_start = float(self.scheduler_config.get("pct_start", 0.3))
            div_factor = float(self.scheduler_config.get("div_factor", 25.0))
            final_div_factor = float(self.scheduler_config.get("final_div_factor", 1e4))
            
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=max_lr,
                total_steps=total_steps,
                pct_start=pct_start,
                div_factor=div_factor,
                final_div_factor=final_div_factor
            )
            print(f"Using OneCycleLR: max_lr={max_lr}, total_steps={total_steps}, pct_start={pct_start}")
            
        else:
            raise ValueError(f"Unsupported scheduler_type: {self.scheduler_type}. Supported: 'cosine', 'cosine_warm_restarts', 'onecycle'")
        
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
        a_prime_t2_hat: torch.Tensor,            # Model's predicted t2 action
        at2_6dof_gt: torch.Tensor                # Ground truth at2_6dof from JSON (auxiliary task target)
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Computes losses with configurable weights for main and auxiliary tasks.
        
        Main task: Predict at1_6dof from image and action change (predicted_action_composed vs at1_6dof_gt)
        Auxiliary task: t2 action prediction (a_prime_t2_hat vs at2_6dof_gt)
        
        Note: Removed latent reconstruction loss as it was comparing t1 features incorrectly
        """
        # Main task loss - L_SmoothL1(predicted_action_composed, at1_6dof_gt)
        main_task_loss = self.smooth_l1_loss(predicted_action_composed, at1_6dof_gt)
        
        # Auxiliary task loss - help with training process
        aux_t2_action_loss = self.smooth_l1_loss(a_prime_t2_hat, at2_6dof_gt)
        
        # Combine losses based on configuration
        if self.hparams.primary_task_only:
            # Only use main task loss - corresponds to evaluation mode
            total_loss = main_task_loss
        else:
            # Full training loss - includes auxiliary task to help training
            total_loss = main_task_loss + self.lambda_t2_action * aux_t2_action_loss
        
        loss_dict = {
            "total_loss": total_loss,
            "main_task_loss": main_task_loss,  # This is the main metric we care about
            "aux_t2_action_loss": aux_t2_action_loss
        }
        return total_loss, loss_dict
    
    def training_step(self, batch: Tuple[torch.Tensor, ...], batch_idx: int) -> torch.Tensor:
        # Batch structure: (image_t1, a_hat_t1_to_t2_gt, at1_6dof_gt, at2_6dof_gt)
        image_t1, a_hat_t1_to_t2_gt, at1_6dof_gt, at2_6dof_gt = batch
        
        system_outputs = self(image_t1, a_hat_t1_to_t2_gt)
        
        predicted_composed_action = system_outputs["predicted_action_composed"]
        a_prime_t2_hat = system_outputs["a_prime_t2_hat"]
        
        total_loss, loss_dict = self.compute_losses(
            predicted_composed_action, 
            at1_6dof_gt,  # Main task: predict at1_6dof
            a_prime_t2_hat,
            at2_6dof_gt   # Auxiliary task: predict at2_6dof
        )
        
        # Step-level logging (x: global_step) - used for debugging
        self.log("train_total_loss_step", total_loss, prog_bar=True, on_step=True, on_epoch=False)
        self.log("train_main_task_loss_step", loss_dict["main_task_loss"], prog_bar=True, on_step=True, on_epoch=False)

        # Epoch-level logging (x: epoch) - used for clear epoch trends
        # Check if trainer is available (not available during standalone testing)
        try:
            if self.trainer is not None:
                self.log("train_total_loss_epoch", total_loss, on_step=False, on_epoch=True)
                self.log("train_main_task_loss_epoch", loss_dict["main_task_loss"], on_step=False, on_epoch=True)
                self.log("train_aux_t2_loss", loss_dict["aux_t2_action_loss"], on_step=False, on_epoch=True)
            else:
                # Standalone testing - use simple logging without step parameter
                self.log("train_total_loss_epoch", total_loss, on_step=False, on_epoch=True)
                self.log("train_main_task_loss_epoch", loss_dict["main_task_loss"], on_step=False, on_epoch=True)
                self.log("train_aux_t2_loss", loss_dict["aux_t2_action_loss"], on_step=False, on_epoch=True)
        except (RuntimeError, AttributeError):
            # Fallback for standalone testing
            self.log("train_total_loss_epoch", total_loss, on_step=False, on_epoch=True)
            self.log("train_main_task_loss_epoch", loss_dict["main_task_loss"], on_step=False, on_epoch=True)
            self.log("train_aux_t2_loss", loss_dict["aux_t2_action_loss"], on_step=False, on_epoch=True)
        return total_loss
    
    def validation_step(self, batch: Tuple[torch.Tensor, ...], batch_idx: int) -> torch.Tensor:
        image_t1, a_hat_t1_to_t2_gt, at1_6dof_gt, at2_6dof_gt = batch
        system_outputs = self(image_t1, a_hat_t1_to_t2_gt)
        
        predicted_composed_action = system_outputs["predicted_action_composed"]
        a_prime_t2_hat = system_outputs["a_prime_t2_hat"]
            
        total_loss, loss_dict = self.compute_losses(
            predicted_composed_action, 
            at1_6dof_gt,  # Main task: predict at1_6dof
            a_prime_t2_hat,
            at2_6dof_gt   # Auxiliary task: predict at2_6dof
        )
        
        # Validation logging (x: epoch) - validation is epoch-level
        # Check if trainer is available (not available during standalone testing)
        try:
            if self.trainer is not None:
                self.log("val_total_loss", total_loss, prog_bar=True, on_step=False, on_epoch=True)
                self.log("val_main_task_loss", loss_dict["main_task_loss"], prog_bar=True, on_step=False, on_epoch=True)
                self.log("val_aux_t2_loss", loss_dict["aux_t2_action_loss"], on_step=False, on_epoch=True)
            else:
                # Standalone testing - use simple logging without step parameter
                self.log("val_total_loss", total_loss, prog_bar=True, on_step=False, on_epoch=True)
                self.log("val_main_task_loss", loss_dict["main_task_loss"], prog_bar=True, on_step=False, on_epoch=True)
                self.log("val_aux_t2_loss", loss_dict["aux_t2_action_loss"], on_step=False, on_epoch=True)
        except (RuntimeError, AttributeError):
            # Fallback for standalone testing
            self.log("val_total_loss", total_loss, prog_bar=True, on_step=False, on_epoch=True)
            self.log("val_main_task_loss", loss_dict["main_task_loss"], prog_bar=True, on_step=False, on_epoch=True)
            self.log("val_aux_t2_loss", loss_dict["aux_t2_action_loss"], on_step=False, on_epoch=True)
        
        # 🔧 修復記憶體洩漏：限制累積數量並確保tensor移到CPU
        if len(self.validation_step_outputs) < self.max_validation_outputs:
            # Collect predictions and ground truth for scatter plots
            self.validation_step_outputs.append({
                "predicted_action_composed": predicted_composed_action.detach().cpu().clone(),
                "target_action_composed_gt": at1_6dof_gt.detach().cpu().clone()
            })
        
        # 🔧 定期清理GPU記憶體
        if batch_idx % 50 == 0:  # 每50個batch清理一次
            torch.cuda.empty_cache()
        
        return total_loss
    
    def test_step(self, batch: Tuple[torch.Tensor, ...], batch_idx: int) -> Dict[str, torch.Tensor]:
        image_t1, a_hat_t1_to_t2_gt, at1_6dof_gt, at2_6dof_gt = batch
        system_outputs = self(image_t1, a_hat_t1_to_t2_gt)
        
        predicted_composed_action = system_outputs["predicted_action_composed"]
        a_prime_t2_hat = system_outputs["a_prime_t2_hat"]
            
        total_loss, loss_dict = self.compute_losses(
            predicted_composed_action, 
            at1_6dof_gt,  # Main task: predict at1_6dof
            a_prime_t2_hat,
            at2_6dof_gt   # Auxiliary task: predict at2_6dof
        )
        
        # For reporting, calculate MSE of the primary composed action prediction
        mse_main_task = F.mse_loss(predicted_composed_action, at1_6dof_gt)
        
        # Test logging (橫軸: epoch)
        # Check if trainer is available (not available during standalone testing)
        try:
            if self.trainer is not None:
                self.log("test_total_loss", total_loss, on_step=False, on_epoch=True)
                self.log("test_main_task_loss", loss_dict["main_task_loss"], on_step=False, on_epoch=True)
                self.log("test_main_task_mse", mse_main_task, on_step=False, on_epoch=True)
                self.log("test_aux_t2_loss", loss_dict["aux_t2_action_loss"], on_step=False, on_epoch=True)
            else:
                # Standalone testing - use simple logging without step parameter
                self.log("test_total_loss", total_loss, on_step=False, on_epoch=True)
                self.log("test_main_task_loss", loss_dict["main_task_loss"], on_step=False, on_epoch=True)
                self.log("test_main_task_mse", mse_main_task, on_step=False, on_epoch=True)
                self.log("test_aux_t2_loss", loss_dict["aux_t2_action_loss"], on_step=False, on_epoch=True)
        except (RuntimeError, AttributeError):
            # Fallback for standalone testing
            self.log("test_total_loss", total_loss, on_step=False, on_epoch=True)
            self.log("test_main_task_loss", loss_dict["main_task_loss"], on_step=False, on_epoch=True)
            self.log("test_main_task_mse", mse_main_task, on_step=False, on_epoch=True)
            self.log("test_aux_t2_loss", loss_dict["aux_t2_action_loss"], on_step=False, on_epoch=True)
        
        # 🔧 freeze memory
        if len(self.test_step_outputs) < self.max_test_outputs:
            self.test_step_outputs.append({
                "test_total_loss": total_loss.detach().cpu().clone(),
                "test_main_task_mse": mse_main_task.detach().cpu().clone(),
                "predicted_action_composed": predicted_composed_action.detach().cpu().clone(),
                "target_action_composed_gt": at1_6dof_gt.detach().cpu().clone(),
                "a_prime_t2_hat": a_prime_t2_hat.detach().cpu().clone(),
                "target_a_t2_prime_gt": at2_6dof_gt.detach().cpu().clone()
            })
        
        # 🔧 freeze memory
        if batch_idx % 50 == 0:  # clear cache every 50 batches
            torch.cuda.empty_cache()
        
        return {
            "test_total_loss": total_loss.detach().cpu(),
            "test_main_task_mse": mse_main_task.detach().cpu(),
            "predicted_action_composed": predicted_composed_action.detach().cpu(),
            "target_action_composed_gt": at1_6dof_gt.detach().cpu(),
            "a_prime_t2_hat": a_prime_t2_hat.detach().cpu(),
            "target_a_t2_prime_gt": at2_6dof_gt.detach().cpu()
        }
    
    def on_test_epoch_end(self):
        """PyTorch Lightning v2.0.0 compatible test epoch end hook"""
        if not self.test_step_outputs:
            return
            
        all_preds_composed = torch.cat([out["predicted_action_composed"] for out in self.test_step_outputs])
        all_targets_composed_gt = torch.cat([out["target_action_composed_gt"] for out in self.test_step_outputs])
        
        overall_mse_main_task = F.mse_loss(all_preds_composed, all_targets_composed_gt)
        # Check if trainer is available (not available during standalone testing)
        try:
            if self.trainer is not None:
                self.log("test_final_main_task_mse", overall_mse_main_task)
            else:
                self.log("test_final_main_task_mse", overall_mse_main_task)
        except (RuntimeError, AttributeError):
            self.log("test_final_main_task_mse", overall_mse_main_task)

        per_dim_mse_main_task = torch.mean((all_preds_composed - all_targets_composed_gt) ** 2, dim=0)
        dim_names = ["X", "Y", "Z", "Roll", "Pitch", "Yaw"]
        for i, dim_mse in enumerate(per_dim_mse_main_task):
            try:
                if self.trainer is not None:
                    self.log(f"test_main_task_mse_{dim_names[i]}", dim_mse)
                else:
                    self.log(f"test_main_task_mse_{dim_names[i]}", dim_mse)
            except (RuntimeError, AttributeError):
                self.log(f"test_main_task_mse_{dim_names[i]}", dim_mse)
        
        # 🔧 freeze memory      
        self.test_step_outputs.clear()
        del all_preds_composed, all_targets_composed_gt
        gc.collect()
        torch.cuda.empty_cache()

    def on_validation_epoch_end(self):
        """Collect validation data without generating plots (plots generated at training end)"""
        if not self.validation_step_outputs:
            return
        
        
        # only keep the last epoch's data for final plot
        current_epoch = getattr(self, 'current_epoch', 0)
        
        # save the current epoch's data to model attribute (for final plot after training)
        all_preds = torch.cat([out["predicted_action_composed"] for out in self.validation_step_outputs])
        all_targets = torch.cat([out["target_action_composed_gt"] for out in self.validation_step_outputs])
        
        # save the latest validation data to model attribute
        self.latest_validation_data = {
            "predictions": all_preds.clone(),
            "targets": all_targets.clone(),
            "epoch": current_epoch
        }
        
        print(f"[VALIDATION] Validation epoch {current_epoch}: collected {len(all_preds)} samples' predictions")
        print(f"    will generate scatter plots after training")
        
        # 🔧 freeze memory
        self.validation_step_outputs.clear()
        del all_preds, all_targets
        gc.collect()
        torch.cuda.empty_cache()
    
    def generate_final_validation_plots(self, output_dir: str = None):
        """Generate validation scatter plots at the end of training"""
        if not hasattr(self, 'latest_validation_data') or self.latest_validation_data is None:
            print("[WARNING] no validation data available for generating scatter plots")
            return
        
        print("[VALIDATION] generating final validation scatter plots...")
        
        # get saved validation data
        all_preds = self.latest_validation_data["predictions"]
        all_targets = self.latest_validation_data["targets"]
        final_epoch = self.latest_validation_data["epoch"]
        
        # Convert to numpy for plotting
        preds_np = all_preds.numpy()
        targets_np = all_targets.numpy()
        
        # Create scatter plots for each 6DOF dimension
        dim_names = ["X", "Y", "Z", "Roll", "Pitch", "Yaw"]
        
        # Create output directory for plots
        if output_dir is None:
            if hasattr(self, 'trainer') and self.trainer and hasattr(self.trainer, 'logger'):
                if hasattr(self.trainer.logger, 'log_dir'):
                    plots_dir = os.path.join(self.trainer.logger.log_dir, "final_validation_plots")
                else:
                    plots_dir = "final_validation_plots"
            else:
                plots_dir = "final_validation_plots"
        else:
            plots_dir = os.path.join(output_dir, "final_validation_plots")
        
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
            plt.title(f'Final Validation: Predicted vs Ground Truth - {dim_name}\nCorr: {correlation:.3f}, MSE: {mse:.4f}', fontsize=14)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Make plot square
            plt.axis('equal')
            
            # Save plot
            plot_path = os.path.join(plots_dir, f'final_validation_scatter_{dim_name.lower()}.png')
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
        
        plt.suptitle(f'Final Validation Results - Epoch {final_epoch} ({len(all_preds)} samples)', fontsize=16)
        plt.tight_layout()
        
        # Save combined plot
        combined_plot_path = os.path.join(plots_dir, f'final_validation_scatter_combined.png')
        plt.savefig(combined_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"[VALIDATION] final validation scatter plots saved to: {plots_dir}")
        print(f"   - individual plots: final_validation_scatter_[dimension].png")
        print(f"   - combined plot: final_validation_scatter_combined.png")
        print(f"   - based on epoch {final_epoch} with {len(all_preds)} samples")
        
        # 🔧 freeze memory
        del all_preds, all_targets, preds_np, targets_np
        self.latest_validation_data = None  # clear saved data
        gc.collect()
        torch.cuda.empty_cache()

def get_cardiac_dreamer_system(
    d_model: int = 768,
    nhead: int = 12,
    num_layers: int = 6,
    feature_dim: int = 49, 
    lr: float = 1e-4,
    weight_decay: float = 1e-5,
    lambda_t2_action: float = 1.0,
    smooth_l1_beta: float = 1.0,
    use_flash_attn: bool = True,
    in_channels: int = 1, 
    use_pretrained: bool = True,
    primary_task_only: bool = False,
    freeze_backbone_layers: int = 0,
    scheduler_type: str = "cosine",
    scheduler_config: dict = None
) -> CardiacDreamerSystem:
    return CardiacDreamerSystem(
        token_type="channel",
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        feature_dim=feature_dim,
        lr=lr,
        weight_decay=weight_decay,
        lambda_t2_action=lambda_t2_action,
        smooth_l1_beta=smooth_l1_beta,
        use_flash_attn=use_flash_attn,
        in_channels=in_channels, 
        use_pretrained=use_pretrained,
        primary_task_only=primary_task_only,
        freeze_backbone_layers=freeze_backbone_layers,
        scheduler_type=scheduler_type,
        scheduler_config=scheduler_config
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
    LAMBDA_T2_ACTION = 1.0
    SMOOTH_L1_BETA = 1.0
    USE_FLASH_ATTN = False
    IN_CHANNELS = 1
    USE_PRETRAINED = True
    FREEZE_BACKBONE_LAYERS = 0
    SCHEDULER_TYPE = "cosine"
    SCHEDULER_CONFIG = None

    system = get_cardiac_dreamer_system(
        token_type=TOKEN_TYPE,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        feature_dim=FEATURE_DIM,
        lr=LR,
        lambda_t2_action=LAMBDA_T2_ACTION,
        smooth_l1_beta=SMOOTH_L1_BETA,
        use_flash_attn=USE_FLASH_ATTN,
        in_channels=IN_CHANNELS,
        use_pretrained=USE_PRETRAINED,
        primary_task_only=False,
        freeze_backbone_layers=FREEZE_BACKBONE_LAYERS,
        scheduler_type=SCHEDULER_TYPE,
        scheduler_config=SCHEDULER_CONFIG
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
        a_prime_t2_hat = system_outputs["a_prime_t2_hat"]
        
        print(f"\nSystem output shapes:")
        print(f"  Predicted composed action (a_t1_prime_composed): {predicted_action_composed.shape}")
        print(f"  Predicted action at t2 (a_prime_t2_hat): {a_prime_t2_hat.shape}")

        total_loss, loss_dict = system.compute_losses(
            predicted_action_composed,
            at1_6dof_gt, 
            a_prime_t2_hat,
            at2_6dof_gt 
        )
        
        print(f"\nLoss values:")
        print(f"  Total loss: {total_loss.item():.4f}")
        print(f"  Main Task Loss: {loss_dict['main_task_loss'].item():.4f}")
        print(f"  Aux T2 Action Loss: {loss_dict['aux_t2_action_loss'].item():.4f}")

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