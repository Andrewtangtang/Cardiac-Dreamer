"""
Training callbacks for cardiac dreamer
"""

import os
from typing import Dict, List
from pytorch_lightning.callbacks import (
    ModelCheckpoint, 
    EarlyStopping, 
    LearningRateMonitor, 
    DeviceStatsMonitor
)


def setup_callbacks(output_dir: str, train_config: Dict) -> List:
    """Set up training callbacks"""
    callbacks = []
    
    # Model checkpoint - save best models
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(output_dir, "checkpoints"),
        filename="cardiac_dreamer-{epoch:02d}-{val_main_task_loss:.4f}",
        save_top_k=3,
        monitor="val_main_task_loss",  # Monitor main task loss
        mode="min",
        save_last=True,
        save_weights_only=False,
        verbose=True
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping
    early_stop_callback = EarlyStopping(
        monitor="val_main_task_loss",
        patience=train_config["early_stop_patience"],
        mode="min",
        verbose=True,
        strict=False
    )
    callbacks.append(early_stop_callback)
    
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(
        logging_interval="epoch",
        log_momentum=True
    )
    callbacks.append(lr_monitor)
    
    # Device stats monitor
    device_stats = DeviceStatsMonitor()
    callbacks.append(device_stats)
    
    return callbacks 