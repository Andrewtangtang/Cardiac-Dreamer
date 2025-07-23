"""
Training loggers for cardiac dreamer
"""

import os
from datetime import datetime
from typing import Dict, List
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger


def setup_loggers(output_dir: str, config_override: Dict = None) -> List:
    """Set up training loggers"""
    loggers = []
    
    # TensorBoard logger
    tb_logger = TensorBoardLogger(
        save_dir=os.path.join(output_dir, "logs"),
        name="cardiac_dreamer",
        default_hp_metric=False
    )
    loggers.append(tb_logger)
    
    # CSV logger for easy analysis
    csv_logger = CSVLogger(
        save_dir=os.path.join(output_dir, "logs"),
        name="cardiac_dreamer_csv"
    )
    loggers.append(csv_logger)
    
    # WandB logger (if enabled)
    if config_override and config_override.get("experiment", {}).get("use_wandb", False):
        try:
            from pytorch_lightning.loggers import WandbLogger
            
            experiment_config = config_override.get("experiment", {})
            
            # 簡化配置，避免權限問題
            wandb_config = {
                "project": experiment_config.get("wandb_project", "cardiac_dreamer"),
                "tags": experiment_config.get("tags", []),
                "notes": experiment_config.get("description", ""),
                "name": f"cardiac_dreamer_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            }
            
            # 添加 entity（如果有指定）
            entity = experiment_config.get("wandb_entity")
            if entity:
                wandb_config["entity"] = entity
                
            wandb_logger = WandbLogger(**wandb_config)
            loggers.append(wandb_logger)
            print(f"✅ WandB logger enabled - Project: {experiment_config.get('wandb_project')}")
            
        except ImportError:
            print("⚠️ WandB not available, skipping WandB logger")
        except Exception as e:
            print(f"⚠️ Failed to setup WandB logger: {e}")
            print("   Continuing without WandB...")
    
    return loggers 