"""
Learning rate schedulers and advanced training strategies
"""

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, StepLR
from typing import Dict, Any, Optional
import pytorch_lightning as pl


class EnhancedReduceLROnPlateau(ReduceLROnPlateau):
    """
    Enhanced ReduceLROnPlateau with additional features
    """
    
    def __init__(self, optimizer, mode='min', factor=0.5, patience=5, 
                 threshold=1e-4, threshold_mode='rel', cooldown=0, min_lr=1e-7, eps=1e-8):
        super().__init__(optimizer, mode, factor, patience, threshold, 
                        threshold_mode, cooldown, min_lr, eps)
        self.best_score = None
        self.num_bad_epochs = 0
        
    def step(self, metrics, epoch=None):
        """Step with enhanced logging"""
        current = float(metrics)
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0

        if self.num_bad_epochs > self.patience:
            self._reduce_lr(epoch)
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0

    def get_lr_info(self):
        """Get current learning rate information"""
        return {
            'current_lr': [group['lr'] for group in self.optimizer.param_groups],
            'num_bad_epochs': self.num_bad_epochs,
            'patience': self.patience,
            'best_score': self.best
        }


class WarmupScheduler:
    """
    Learning rate warmup scheduler
    
    Args:
        optimizer: Optimizer to schedule
        warmup_epochs: Number of epochs for warmup
        base_scheduler: Base scheduler to use after warmup
    """
    
    def __init__(self, optimizer, warmup_epochs: int = 5, base_scheduler=None):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.base_scheduler = base_scheduler
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.current_epoch = 0
        
    def step(self, metrics=None):
        """Step the scheduler"""
        if self.current_epoch < self.warmup_epochs:
            # Warmup phase
            lr_scale = (self.current_epoch + 1) / self.warmup_epochs
            for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                param_group['lr'] = base_lr * lr_scale
        else:
            # Use base scheduler
            if self.base_scheduler:
                if metrics is not None:
                    self.base_scheduler.step(metrics)
                else:
                    self.base_scheduler.step()
                    
        self.current_epoch += 1
        
    def state_dict(self):
        """Get scheduler state"""
        state = {
            'current_epoch': self.current_epoch,
            'base_lrs': self.base_lrs
        }
        if self.base_scheduler:
            state['base_scheduler'] = self.base_scheduler.state_dict()
        return state
        
    def load_state_dict(self, state_dict):
        """Load scheduler state"""
        self.current_epoch = state_dict['current_epoch']
        self.base_lrs = state_dict['base_lrs']
        if self.base_scheduler and 'base_scheduler' in state_dict:
            self.base_scheduler.load_state_dict(state_dict['base_scheduler'])


def create_scheduler(optimizer, config: Dict[str, Any]):
    """
    Create learning rate scheduler based on configuration
    
    Args:
        optimizer: Optimizer to schedule
        config: Configuration dictionary
        
    Returns:
        Scheduler instance or None
    """
    regularization_config = config.get('regularization', {})
    lr_config = regularization_config.get('lr_scheduler', {})
    
    if not lr_config:
        return None
        
    scheduler_type = lr_config.get('type', 'reduce_on_plateau')
    
    if scheduler_type == 'reduce_on_plateau':
        scheduler = EnhancedReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=lr_config.get('factor', 0.5),
            patience=lr_config.get('patience', 5),
            min_lr=lr_config.get('min_lr', 1e-7),
            threshold=lr_config.get('threshold', 1e-4)
        )
        
    elif scheduler_type == 'cosine_annealing':
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=lr_config.get('t_max', 50),
            eta_min=lr_config.get('min_lr', 1e-7)
        )
        
    elif scheduler_type == 'step':
        scheduler = StepLR(
            optimizer,
            step_size=lr_config.get('step_size', 10),
            gamma=lr_config.get('gamma', 0.5)
        )
        
    else:
        print(f"Unknown scheduler type: {scheduler_type}")
        return None
    
    # Add warmup if specified
    warmup_epochs = lr_config.get('warmup_epochs', 0)
    if warmup_epochs > 0:
        scheduler = WarmupScheduler(
            optimizer,
            warmup_epochs=warmup_epochs,
            base_scheduler=scheduler
        )
        
    return scheduler


class GradientClipping:
    """
    Enhanced gradient clipping with monitoring
    
    Args:
        max_norm: Maximum gradient norm
        norm_type: Type of norm to use
        error_if_nonfinite: Whether to error on non-finite gradients
    """
    
    def __init__(self, max_norm: float = 1.0, norm_type: float = 2.0, error_if_nonfinite: bool = True):
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.error_if_nonfinite = error_if_nonfinite
        self.gradient_norms = []
        
    def clip_gradients(self, model: nn.Module):
        """Clip gradients and return gradient norm"""
        total_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=self.max_norm,
            norm_type=self.norm_type,
            error_if_nonfinite=self.error_if_nonfinite
        )
        
        self.gradient_norms.append(float(total_norm))
        return total_norm
        
    def get_gradient_stats(self):
        """Get gradient norm statistics"""
        if not self.gradient_norms:
            return {}
            
        import numpy as np
        return {
            'mean_grad_norm': np.mean(self.gradient_norms[-100:]),  # Last 100 steps
            'max_grad_norm': np.max(self.gradient_norms[-100:]),
            'min_grad_norm': np.min(self.gradient_norms[-100:]),
            'current_grad_norm': self.gradient_norms[-1] if self.gradient_norms else 0
        }


class TrainingStepLogger:
    """
    Enhanced training step logging
    """
    
    def __init__(self, log_frequency: int = 100):
        self.log_frequency = log_frequency
        self.step_count = 0
        self.losses = []
        self.learning_rates = []
        
    def log_step(self, loss: float, lr: float, additional_metrics: Dict[str, float] = None):
        """Log training step"""
        self.step_count += 1
        self.losses.append(loss)
        self.learning_rates.append(lr)
        
        if self.step_count % self.log_frequency == 0:
            self._print_step_info(additional_metrics or {})
            
    def _print_step_info(self, additional_metrics: Dict[str, float]):
        """Print step information"""
        import numpy as np
        recent_loss = np.mean(self.losses[-self.log_frequency:])
        current_lr = self.learning_rates[-1]
        
        info_str = f"Step {self.step_count}: Loss={recent_loss:.6f}, LR={current_lr:.2e}"
        
        for metric_name, metric_value in additional_metrics.items():
            info_str += f", {metric_name}={metric_value:.4f}"
            
        print(info_str)


def create_training_components(config: Dict[str, Any], optimizer, model: nn.Module):
    """
    Create enhanced training components based on configuration
    
    Args:
        config: Configuration dictionary
        optimizer: Optimizer instance
        model: Model instance
        
    Returns:
        Dictionary with training components
    """
    components = {}
    
    # Learning rate scheduler
    scheduler = create_scheduler(optimizer, config)
    if scheduler:
        components['scheduler'] = scheduler
        
    # Gradient clipping
    train_config = config.get('training', {})
    clip_val = train_config.get('gradient_clip_val', 0)
    if clip_val > 0:
        components['gradient_clipper'] = GradientClipping(max_norm=clip_val)
        
    # Training logger
    log_frequency = train_config.get('log_every_n_steps', 100)
    components['step_logger'] = TrainingStepLogger(log_frequency=log_frequency)
    
    return components 