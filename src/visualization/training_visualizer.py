"""
Training visualization utilities for cardiac dreamer
"""

import os
import json
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pytorch_lightning.loggers import CSVLogger
from typing import Dict


class TrainingVisualizer:
    """Handle training visualization and logging"""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.plots_dir = os.path.join(output_dir, "plots")
        os.makedirs(self.plots_dir, exist_ok=True)
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def plot_dataset_statistics(self, train_dataset, val_dataset, test_dataset):
        """Plot dataset statistics"""
        print("Creating dataset visualization...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Dataset sizes
        if test_dataset:
            sizes = [len(train_dataset), len(val_dataset), len(test_dataset)]
            labels = ['Train', 'Validation', 'Test']
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        else:
            sizes = [len(train_dataset), len(val_dataset)]
            labels = ['Train', 'Validation']
            colors = ['#FF6B6B', '#4ECDC4']
        
        axes[0, 0].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        axes[0, 0].set_title('Dataset Distribution')
        
        # Patient distribution
        train_stats = train_dataset.get_patient_stats()
        val_stats = val_dataset.get_patient_stats()
        test_stats = test_dataset.get_patient_stats() if test_dataset else {}
        
        # Bar plot for patient statistics
        patients = list(set(list(train_stats.keys()) + list(val_stats.keys()) + list(test_stats.keys())))
        train_counts = [train_stats.get(p, 0) for p in patients]
        val_counts = [val_stats.get(p, 0) for p in patients]
        
        if test_dataset:
            test_counts = [test_stats.get(p, 0) for p in patients]
            x = np.arange(len(patients))
            width = 0.25
            
            axes[0, 1].bar(x - width, train_counts, width, label='Train', color=colors[0])
            axes[0, 1].bar(x, val_counts, width, label='Val', color=colors[1])
            axes[0, 1].bar(x + width, test_counts, width, label='Test', color=colors[2])
        else:
            x = np.arange(len(patients))
            width = 0.35
            
            axes[0, 1].bar(x - width/2, train_counts, width, label='Train', color=colors[0])
            axes[0, 1].bar(x + width/2, val_counts, width, label='Val', color=colors[1])
        
        axes[0, 1].set_xlabel('Patients')
        axes[0, 1].set_ylabel('Number of Samples')
        axes[0, 1].set_title('Samples per Patient')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(patients, rotation=45)
        axes[0, 1].legend()
        
        # Sample a few images for visualization
        sample_images = []
        sample_labels = []
        for i in range(min(4, len(train_dataset))):
            img, _, _, _ = train_dataset[i]
            sample_images.append(img.squeeze().numpy())
            sample_labels.append(f"Sample {i+1}")
        
        for i, (img, label) in enumerate(zip(sample_images, sample_labels)):
            row, col = (i // 2) + 1, i % 2
            if row < 2:
                axes[row, col].imshow(img, cmap='gray')
                axes[row, col].set_title(label)
                axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'dataset_statistics.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save statistics to file
        stats_file = os.path.join(self.output_dir, 'dataset_stats.json')
        stats = {
            'total_samples': {
                'train': len(train_dataset),
                'val': len(val_dataset),
                'test': len(test_dataset) if test_dataset else 0
            },
            'patient_distribution': {
                'train': train_stats,
                'val': val_stats,
                'test': test_stats if test_dataset else {}
            }
        }
        if not test_dataset:
            stats['note'] = 'No test set (custom split with patients 1-5 as validation)'
        
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"Dataset statistics saved to {self.plots_dir}")
    
    def create_training_summary(self, trainer, model):
        """Create training summary plots"""
        print("Creating training summary...")
        
        # Get training logs
        csv_logger = None
        for logger in trainer.loggers:
            if isinstance(logger, CSVLogger):
                csv_logger = logger
                break
        
        if csv_logger is None:
            print("No CSV logger found, skipping training plots")
            return
        
        # Read logs
        log_file = os.path.join(csv_logger.log_dir, "metrics.csv")
        if not os.path.exists(log_file):
            print("No metrics file found, skipping training plots")
            return
        
        df = pd.read_csv(log_file)
        
        # Print available columns for debugging
        print(f"Available columns in CSV: {list(df.columns)}")
        
        # Create simplified training plots focusing on essential losses (epoch-based)
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Training and validation total loss (epoch-based)
        train_loss = df[df['train_total_loss_epoch'].notna()]
        val_loss = df[df['val_total_loss'].notna()]
        
        if not train_loss.empty and not val_loss.empty:
            axes[0, 0].plot(train_loss['epoch'], train_loss['train_total_loss_epoch'], label='Train', color='#FF6B6B', linewidth=2)
            axes[0, 0].plot(val_loss['epoch'], val_loss['val_total_loss'], label='Validation', color='#4ECDC4', linewidth=2)
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Total Loss')
            axes[0, 0].set_title('Training & Validation Total Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # Main task loss (most important metric) (epoch-based)
        train_main = df[df['train_main_task_loss_epoch'].notna()]
        val_main = df[df['val_main_task_loss'].notna()]
        
        if not train_main.empty and not val_main.empty:
            axes[0, 1].plot(train_main['epoch'], train_main['train_main_task_loss_epoch'], label='Train', color='#FF6B6B', linewidth=2)
            axes[0, 1].plot(val_main['epoch'], val_main['val_main_task_loss'], label='Validation', color='#4ECDC4', linewidth=2)
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Main Task Loss')
            axes[0, 1].set_title('Main Task Loss (AT1 Prediction) - Key Metric')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Learning rate (epoch-based)
        lr_data = df[df['lr-AdamW'].notna()]
        if not lr_data.empty:
            axes[0, 2].plot(lr_data['epoch'], lr_data['lr-AdamW'], color='#45B7D1', linewidth=2)
            axes[0, 2].set_xlabel('Epoch')
            axes[0, 2].set_ylabel('Learning Rate')
            axes[0, 2].set_title('Learning Rate Schedule')
            axes[0, 2].grid(True, alpha=0.3)
            axes[0, 2].set_yscale('log')  # Use log scale for learning rate
        
        # Auxiliary t2 action loss (epoch-based)
        try:
            aux_t2_loss = df[df['train_aux_t2_loss'].notna()]
            val_aux_t2_loss = df[df['val_aux_t2_loss'].notna()]
            
            if not aux_t2_loss.empty:
                axes[1, 0].plot(aux_t2_loss['epoch'], aux_t2_loss['train_aux_t2_loss'], label='Train', color='#FECA57', linewidth=2)
                if not val_aux_t2_loss.empty:
                    axes[1, 0].plot(val_aux_t2_loss['epoch'], val_aux_t2_loss['val_aux_t2_loss'], label='Validation', color='#FFD93D', linewidth=2)
                axes[1, 0].set_xlabel('Epoch')
                axes[1, 0].set_ylabel('Auxiliary T2 Loss')
                axes[1, 0].set_title('T2 Action Prediction Loss (Auxiliary)')
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)
            else:
                # 如果沒有輔助損失數據，顯示提示信息
                axes[1, 0].text(0.5, 0.5, 'No auxiliary T2 loss data available', 
                               ha='center', va='center', transform=axes[1, 0].transAxes)
                axes[1, 0].set_title('T2 Action Prediction Loss (Auxiliary)')
                
        except KeyError as e:
            print(f"[WARNING] Column {e} not found in training logs")
            axes[1, 0].text(0.5, 0.5, f'Auxiliary loss data not available\n(Column {e} missing)', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('T2 Action Prediction Loss (Auxiliary)')
        
        # Final metrics summary
        if not val_loss.empty:
            final_metrics = val_loss.iloc[-1]
            
            # Format metrics safely
            epoch_num = int(final_metrics.get('epoch', 0))
            
            total_loss_val = final_metrics.get('val_total_loss', 'N/A')
            if isinstance(total_loss_val, (int, float)):
                total_loss_str = f"{total_loss_val:.4f}"
            else:
                total_loss_str = str(total_loss_val)
            
            main_task_loss_val = final_metrics.get('val_main_task_loss', 'N/A')
            if isinstance(main_task_loss_val, (int, float)):
                main_task_loss_str = f"{main_task_loss_val:.4f}"
            else:
                main_task_loss_str = str(main_task_loss_val)
            
            metrics_text = f"""Final Metrics (Epoch {epoch_num}):

Total Loss: {total_loss_str}

Main Task Loss: {main_task_loss_str}
(This is the key metric for AT1 prediction)

Training completed successfully.
Focus on Main Task Loss for performance.

[CHART] Final validation plots generated at training end"""
            
            axes[1, 1].text(0.1, 0.5, metrics_text, fontsize=11, 
                           verticalalignment='center', fontfamily='monospace')
            axes[1, 1].set_title('Final Training Metrics')
            axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'training_summary.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Training summary saved to {self.plots_dir}")
    
    def create_final_validation_summary(self, output_dir: str):
        """Create a summary of the final validation scatter plots"""
        print("Creating final validation scatter plots summary...")
        
        # Find the final validation plots directory
        final_plots_dir = os.path.join(output_dir, "final_validation_plots")
        
        if not os.path.exists(final_plots_dir):
            print("No final validation plots found, skipping summary")
            return
        
        # Find the combined plot
        combined_plot = os.path.join(final_plots_dir, "final_validation_scatter_combined.png")
        if not os.path.exists(combined_plot):
            print("No combined final validation plot found")
            return
        
        # Copy the plot to the main plots directory for easy access
        summary_plot_path = os.path.join(self.plots_dir, 'final_validation_scatter_plots.png')
        shutil.copy2(combined_plot, summary_plot_path)
        
        print(f"[CHART] Final validation scatter plots copied to: {summary_plot_path}")
        print(f"[FOLDER] All final validation plots available in: {final_plots_dir}")
        
        # Create a summary text file with information about the plots
        summary_info = {
            "final_validation_plots": {
                "directory": final_plots_dir,
                "combined_plot": combined_plot,
                "description": "Final validation scatter plots showing predicted vs ground truth for each 6DOF dimension",
                "dimensions": ["X", "Y", "Z", "Roll", "Pitch", "Yaw"],
                "plot_types": [
                    "Individual plots: final_validation_scatter_[dimension].png",
                    "Combined plot: final_validation_scatter_combined.png"
                ],
                "optimization": "Generated only once at training end (not every epoch) to prevent memory issues"
            }
        }
        
        summary_file = os.path.join(output_dir, 'final_validation_plots_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary_info, f, indent=2)
        
        print(f"[FILE] Final validation plots summary saved to: {summary_file}") 