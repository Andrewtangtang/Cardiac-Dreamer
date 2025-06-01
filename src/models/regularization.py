"""
Model regularization utilities for cardiac dreamer training
"""

import torch
import torch.nn as nn
from collections import OrderedDict
from typing import Dict, Any


class ModelEMA:
    """
    Exponential Moving Average of model parameters
    
    Args:
        model: The model to track
        decay: Decay rate for the moving average
        device: Device to store the EMA model
    """
    
    def __init__(self, model: nn.Module, decay: float = 0.999, device: torch.device = None):
        self.decay = decay
        self.device = device or torch.device('cpu')
        
        # Create EMA model
        self.ema = self._create_ema_model(model)
        self.ema.eval()
        
        # Initialize with current model parameters
        self._copy_parameters(model, self.ema)
        
        self.updates = 0
        
    def _create_ema_model(self, model: nn.Module) -> nn.Module:
        """Create a copy of the model for EMA"""
        # Get model class and state dict
        ema_model = type(model)(**model.hparams)
        ema_model.load_state_dict(model.state_dict())
        ema_model.to(self.device)
        return ema_model
        
    def _copy_parameters(self, source: nn.Module, target: nn.Module):
        """Copy parameters from source to target model"""
        with torch.no_grad():
            for source_param, target_param in zip(source.parameters(), target.parameters()):
                target_param.copy_(source_param.to(self.device))
                
    def update(self, model: nn.Module):
        """Update EMA parameters"""
        self.updates += 1
        decay = self.decay * (1 - torch.exp(torch.tensor(-self.updates / 2000.0)))
        
        with torch.no_grad():
            for ema_param, model_param in zip(self.ema.parameters(), model.parameters()):
                ema_param.mul_(decay).add_(model_param.to(self.device), alpha=1 - decay)
                
    def apply_shadow(self, model: nn.Module):
        """Apply EMA parameters to model"""
        self._copy_parameters(self.ema, model)
        
    def restore(self, model: nn.Module):
        """Restore original parameters to model"""
        # This would require storing original parameters, simplified for now
        pass
        
    def state_dict(self) -> Dict[str, Any]:
        """Get EMA state dict"""
        return {
            'ema_state_dict': self.ema.state_dict(),
            'decay': self.decay,
            'updates': self.updates
        }
        
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load EMA state dict"""
        self.ema.load_state_dict(state_dict['ema_state_dict'])
        self.decay = state_dict['decay']
        self.updates = state_dict['updates']


class DropoutLayer(nn.Module):
    """
    Configurable dropout layer
    
    Args:
        dropout_rate: Dropout probability
        enabled: Whether dropout is enabled
    """
    
    def __init__(self, dropout_rate: float = 0.1, enabled: bool = True):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.enabled = enabled
        
        if self.enabled and self.dropout_rate > 0:
            self.dropout = nn.Dropout(p=self.dropout_rate)
        else:
            self.dropout = nn.Identity()
            
    def forward(self, x):
        return self.dropout(x)


class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing loss for regression tasks
    
    Args:
        smoothing: Smoothing factor (0 = no smoothing, 1 = maximum smoothing)
        base_loss: Base loss function to apply smoothing to
    """
    
    def __init__(self, smoothing: float = 0.1, base_loss: nn.Module = None):
        super().__init__()
        self.smoothing = smoothing
        self.base_loss = base_loss or nn.SmoothL1Loss()
        
    def forward(self, predictions, targets):
        """Apply label smoothing to regression targets"""
        if self.smoothing <= 0:
            return self.base_loss(predictions, targets)
            
        # For regression, we can add small noise to targets
        noise = torch.randn_like(targets) * self.smoothing * targets.std()
        smoothed_targets = targets + noise
        
        return self.base_loss(predictions, smoothed_targets)


class FeatureDropout(nn.Module):
    """
    Feature dropout for attention mechanisms
    
    Args:
        dropout_rate: Dropout rate for features
        spatial_dropout: Whether to apply spatial dropout
    """
    
    def __init__(self, dropout_rate: float = 0.1, spatial_dropout: bool = False):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.spatial_dropout = spatial_dropout
        
        if spatial_dropout:
            self.dropout = nn.Dropout2d(p=dropout_rate)
        else:
            self.dropout = nn.Dropout(p=dropout_rate)
            
    def forward(self, x):
        if self.training and self.dropout_rate > 0:
            return self.dropout(x)
        return x


def add_dropout_to_model(model: nn.Module, dropout_rate: float = 0.3) -> nn.Module:
    """
    Add dropout layers to an existing model
    
    Args:
        model: Model to add dropout to
        dropout_rate: Dropout rate
        
    Returns:
        Model with added dropout layers
    """
    def add_dropout_recursive(module):
        for name, child in module.named_children():
            if isinstance(child, nn.Linear):
                # Add dropout before linear layers
                setattr(module, f"{name}_dropout", DropoutLayer(dropout_rate))
            else:
                add_dropout_recursive(child)
    
    add_dropout_recursive(model)
    return model


def create_regularized_loss(config: Dict[str, Any]) -> nn.Module:
    """
    Create regularized loss function based on configuration
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Regularized loss function
    """
    regularization_config = config.get('regularization', {})
    advanced_config = config.get('advanced', {})
    
    # Base loss
    base_loss = nn.SmoothL1Loss(beta=config.get('model', {}).get('smooth_l1_beta', 1.0))
    
    # Add label smoothing if specified
    label_smoothing = advanced_config.get('label_smoothing', 0.0)
    if label_smoothing > 0:
        return LabelSmoothingLoss(smoothing=label_smoothing, base_loss=base_loss)
    else:
        return base_loss 