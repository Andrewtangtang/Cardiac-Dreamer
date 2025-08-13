"""
Configuration management utilities for cardiac dreamer training
"""

import os
import yaml
import json
from datetime import datetime
from typing import Dict


def load_config(config_path: str) -> Dict:
    """
    Load YAML configuration file
    
    Args:
        config_path: Configuration file path
    
    Returns:
        Configuration dictionary
    """
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            print(f"Loaded configuration from {config_path}")
        return config
    else:
        print(f"Configuration file {config_path} not found, using defaults")
        return {}


def get_model_config(config_override: Dict = None) -> Dict:
    """
    Get model configuration with production settings
    
    Returns:
        Model configuration dictionary
    """
    default_config = {
        "d_model": 768,
        "num_heads": 12,
        "num_layers": 6,
        "feature_dim": 49,
        "in_channels": 1,
        "use_pretrained": True,
        "lr": 1e-4,
        "weight_decay": 1e-5,
        "lambda_t2_action": 1.0,
        "smooth_l1_beta": 1.0,
        "use_flash_attn": False,
        "primary_task_only": False
    }

    if config_override:
        model_override = config_override.get("model", {})
        default_config.update(model_override)
        
        # Ensure numeric parameters are properly converted to float
        numeric_params = ["lr", "weight_decay", "lambda_t2_action", "smooth_l1_beta"]
        for param in numeric_params:
            if param in default_config:
                try:
                    default_config[param] = float(default_config[param])
                except (ValueError, TypeError):
                    print(f"Warning: Could not convert {param} to float, using default value")
                    # Keep the default value if conversion fails

    return default_config


def get_train_config(config_override: Dict = None) -> Dict:
    """
    Get training configuration with production settings
    
    Returns:
        Training configuration dictionary
    """
    default_config = {
        "batch_size": 16,
        "num_workers": 4,
        "max_epochs": 150,
        "early_stop_patience": 20,
        "accelerator": "auto",
        "precision": 32,
        "gradient_clip_val": 1.0,
        "accumulate_grad_batches": 1,
        "check_val_every_n_epoch": 1,
        "log_every_n_steps": 20
    }
    
    if config_override:
        training_override = config_override.get("training", {})
        default_config.update(training_override)
        
        # Ensure numeric parameters are properly converted to appropriate types
        float_params = ["gradient_clip_val"]
        int_params = ["batch_size", "num_workers", "max_epochs", "early_stop_patience", 
                     "precision", "accumulate_grad_batches", "check_val_every_n_epoch", "log_every_n_steps"]
        
        for param in float_params:
            if param in default_config:
                try:
                    default_config[param] = float(default_config[param])
                except (ValueError, TypeError):
                    print(f"Warning: Could not convert {param} to float, using default value")
        
        for param in int_params:
            if param in default_config:
                try:
                    default_config[param] = int(default_config[param])
                except (ValueError, TypeError):
                    print(f"Warning: Could not convert {param} to int, using default value")
    
    return default_config


def save_experiment_config(output_dir: str, args, model_config: Dict, train_config: Dict):
    """Save experiment configuration"""
    config = {
        "experiment_info": {
            "timestamp": datetime.now().isoformat(),
            "data_dir": args.data_dir,
            "output_dir": args.output_dir,
            "manual_splits": args.manual_splits,
            "train_patients": args.train_patients,
            "val_patients": args.val_patients,
            "test_patients": args.test_patients
        },
        "model_config": model_config,
        "training_config": train_config
    }
    
    config_file = os.path.join(output_dir, "experiment_config.json")
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Experiment configuration saved to {config_file}") 