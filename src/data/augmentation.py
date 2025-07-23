"""
Data augmentation utilities for cardiac dreamer training
"""

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import numpy as np
from PIL import Image
import random
from typing import Union, List


class CardiacAugmentation:
    """
    Custom augmentation class for cardiac ultrasound images
    
    Args:
        rotation_range: Maximum rotation in degrees (±rotation_range)
        brightness_range: Brightness variation. If float, range is [1-val, 1+val]. If List[float, float], range is [min, max].
        contrast_range: Contrast variation. If float, range is [1-val, 1+val]. If List[float, float], range is [min, max].
        noise_std: Standard deviation of Gaussian noise to add
        enabled: Whether augmentation is enabled
    """
    
    def __init__(
        self,
        rotation_range: float = 5.0,
        brightness_range: Union[float, List[float]] = 0.1,
        contrast_range: Union[float, List[float]] = 0.1,
        noise_std: float = 0.01,
        enabled: bool = True
    ):
        self.rotation_range = rotation_range
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.noise_std = noise_std
        self.enabled = enabled
        
    def __call__(self, image):
        """Apply augmentation to image"""
        if not self.enabled:
            return image
            
        # Apply augmentations
        if self.rotation_range > 0:
            image = self._apply_rotation(image)
            
        # Check for brightness adjustment
        apply_brightness = False
        if isinstance(self.brightness_range, float) and self.brightness_range > 0:
            apply_brightness = True
        elif isinstance(self.brightness_range, (list, tuple)) and len(self.brightness_range) == 2:
            # Consider it active if it's a list/tuple of 2, even if it's [1.0, 1.0]
            # The _apply_brightness method will handle the actual factor generation
            apply_brightness = True 
            
        if apply_brightness:
            image = self._apply_brightness(image)

        # Check for contrast adjustment
        apply_contrast = False
        if isinstance(self.contrast_range, float) and self.contrast_range > 0:
            apply_contrast = True
        elif isinstance(self.contrast_range, (list, tuple)) and len(self.contrast_range) == 2:
            apply_contrast = True

        if apply_contrast:
            image = self._apply_contrast(image)
            
        if self.noise_std > 0:
            image = self._apply_noise(image)
            
        return image
    
    def _apply_rotation(self, image):
        """Apply random rotation"""
        angle = random.uniform(-self.rotation_range, self.rotation_range)
        if isinstance(image, Image.Image):
            return TF.rotate(image, angle, fill=0)
        else:  # torch.Tensor
            return TF.rotate(image, angle, fill=0)
    
    def _apply_brightness(self, image):
        """Apply random brightness adjustment"""
        if isinstance(self.brightness_range, float):
            factor = 1.0 + random.uniform(-self.brightness_range, self.brightness_range)
        elif isinstance(self.brightness_range, (list, tuple)) and len(self.brightness_range) == 2:
            factor = random.uniform(self.brightness_range[0], self.brightness_range[1])
        else: # Should not happen if checks in __call__ are correct
            return image 
        
        factor = max(0.1, factor)  # Prevent too dark images
        
        if isinstance(image, Image.Image):
            return TF.adjust_brightness(image, factor)
        else:  # torch.Tensor
            return TF.adjust_brightness(image, factor)
    
    def _apply_contrast(self, image):
        """Apply random contrast adjustment"""
        if isinstance(self.contrast_range, float):
            factor = 1.0 + random.uniform(-self.contrast_range, self.contrast_range)
        elif isinstance(self.contrast_range, (list, tuple)) and len(self.contrast_range) == 2:
            factor = random.uniform(self.contrast_range[0], self.contrast_range[1])
        else: # Should not happen
            return image

        factor = max(0.1, factor)  # Prevent too low contrast
        
        if isinstance(image, Image.Image):
            return TF.adjust_contrast(image, factor)
        else:  # torch.Tensor
            return TF.adjust_contrast(image, factor)
    
    def _apply_noise(self, image):
        """Apply Gaussian noise (only for tensor images)"""
        if isinstance(image, torch.Tensor):
            noise = torch.randn_like(image) * self.noise_std
            return torch.clamp(image + noise, 0, 1)
        else:
            # For PIL images, convert to tensor, apply noise, convert back
            to_tensor = transforms.ToTensor()
            to_pil = transforms.ToPILImage()
            
            tensor_image = to_tensor(image)
            noise = torch.randn_like(tensor_image) * self.noise_std
            noisy_tensor = torch.clamp(tensor_image + noise, 0, 1)
            return to_pil(noisy_tensor)


class MixUp:
    """
    MixUp augmentation for training
    
    Args:
        alpha: Beta distribution parameter for mixing coefficient
        enabled: Whether MixUp is enabled
    """
    
    def __init__(self, alpha: float = 0.2, enabled: bool = True):
        self.alpha = alpha
        self.enabled = enabled
        
    def __call__(self, batch_x, batch_y):
        """Apply MixUp to a batch"""
        if not self.enabled or self.alpha <= 0:
            return batch_x, batch_y, None
            
        batch_size = batch_x.size(0)
        
        # Sample mixing coefficient
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
            
        # Sample random permutation
        index = torch.randperm(batch_size).to(batch_x.device)
        
        # Mix inputs
        mixed_x = lam * batch_x + (1 - lam) * batch_x[index]
        
        # Mix targets (for multiple target types)
        if isinstance(batch_y, (list, tuple)):
            mixed_y = []
            for y in batch_y:
                mixed_y.append(lam * y + (1 - lam) * y[index])
            mixed_y = tuple(mixed_y)
        else:
            mixed_y = lam * batch_y + (1 - lam) * batch_y[index]
            
        return mixed_x, mixed_y, lam


def create_augmented_transform(config):
    """
    Create augmented transform pipeline based on configuration
    
    Args:
        config: Configuration dictionary with augmentation settings
        
    Returns:
        Transform pipeline
    """
    augmentation_config = config.get('data', {}).get('augmentation', {})
    
    # Create base transforms
    transforms_list = [
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ]
    
    # Add augmentation if enabled
    if augmentation_config.get('enabled', False):
        cardiac_aug = CardiacAugmentation(
            rotation_range=augmentation_config.get('rotation_range', 5.0),
            brightness_range=augmentation_config.get('brightness_range', 0.1),
            contrast_range=augmentation_config.get('contrast_range', 0.1),
            noise_std=augmentation_config.get('noise_std', 0.01),
            enabled=True
        )
        # Insert augmentation before ToTensor (works on PIL images)
        transforms_list.insert(-1, cardiac_aug)
    
    # Add normalization
    transforms_list.append(transforms.Normalize(mean=[0.5], std=[0.5]))
    
    return transforms.Compose(transforms_list)


def create_mixup_augmentation(config):
    """
    Create MixUp augmentation based on configuration
    
    Args:
        config: Configuration dictionary
        
    Returns:
        MixUp instance or None
    """
    advanced_config = config.get('advanced', {})
    
    if advanced_config.get('mixup_alpha', 0) > 0:
        return MixUp(
            alpha=advanced_config['mixup_alpha'],
            enabled=True
        )
    else:
        return None 