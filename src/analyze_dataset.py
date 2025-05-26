#!/usr/bin/env python
"""
Dataset Analysis Script for Cardiac Dreamer
Analyzes patient distribution in training, validation, test sets and action_change_6dof statistics
Uses the same CrossPatientTransitionsDataset as training for consistency
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import argparse
from collections import defaultdict
import torchvision.transforms as transforms

# Add project root to path
current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_file_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import dataset utilities from train.py
from src.train import CrossPatientTransitionsDataset, get_patient_splits


def analyze_patient_splits(data_dir: str) -> Dict:
    """Analyze patient grouping using the same method as training"""
    print("\n=== Patient Split Analysis ===")
    
    # Get automatic split results using the same function as training
    train_patients, val_patients, test_patients = get_patient_splits(data_dir)
    
    # Create datasets using the same class as training (without transforms for analysis)
    print("Creating datasets using CrossPatientTransitionsDataset...")
    
    # Simple transform for analysis (just resize, no normalization)
    simple_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    train_dataset = CrossPatientTransitionsDataset(
        data_dir=data_dir,
        transform=simple_transform,
        split="train",
        train_patients=train_patients,
        val_patients=val_patients,
        test_patients=test_patients,
        small_subset=False,
        normalize_actions=False  # Don't normalize for analysis
    )
    
    val_dataset = CrossPatientTransitionsDataset(
        data_dir=data_dir,
        transform=simple_transform,
        split="val",
        train_patients=train_patients,
        val_patients=val_patients,
        test_patients=test_patients,
        small_subset=False,
        normalize_actions=False
    )
    
    test_dataset = CrossPatientTransitionsDataset(
        data_dir=data_dir,
        transform=simple_transform,
        split="test",
        train_patients=train_patients,
        val_patients=val_patients,
        test_patients=test_patients,
        small_subset=False,
        normalize_actions=False
    )
    
    # Get patient statistics from datasets
    train_patient_stats = train_dataset.get_patient_stats()
    val_patient_stats = val_dataset.get_patient_stats()
    test_patient_stats = test_dataset.get_patient_stats()
    
    # Calculate statistics for each split
    splits_info = {
        'train': {
            'patients': list(train_patient_stats.keys()),
            'patient_count': len(train_patient_stats),
            'sample_counts': train_patient_stats,
            'total_samples': sum(train_patient_stats.values())
        },
        'val': {
            'patients': list(val_patient_stats.keys()),
            'patient_count': len(val_patient_stats),
            'sample_counts': val_patient_stats,
            'total_samples': sum(val_patient_stats.values())
        },
        'test': {
            'patients': list(test_patient_stats.keys()),
            'patient_count': len(test_patient_stats),
            'sample_counts': test_patient_stats,
            'total_samples': sum(test_patient_stats.values())
        }
    }
    
    # Print detailed information
    total_patients = len(train_patients) + len(val_patients) + len(test_patients)
    total_samples = splits_info['train']['total_samples'] + splits_info['val']['total_samples'] + splits_info['test']['total_samples']
    
    print(f"\nTotal: {total_patients} patients, {total_samples} samples")
    print("=" * 60)
    
    for split_name, info in splits_info.items():
        print(f"\n{split_name.upper()} Set:")
        print(f"  Number of Patients: {info['patient_count']} ({info['patient_count']/total_patients:.1%})")
        print(f"  Total Samples: {info['total_samples']} ({info['total_samples']/total_samples:.1%})")
        print(f"  Patient List: {info['patients']}")
        print(f"  Samples per Patient:")
        for patient, count in info['sample_counts'].items():
            print(f"    {patient}: {count} samples")
    
    return splits_info, {'train': train_dataset, 'val': val_dataset, 'test': test_dataset}


def extract_action_changes_from_datasets(datasets: Dict) -> np.ndarray:
    """Extract all action_change_6dof data from datasets"""
    print("\n=== Extracting Action Change Data from Datasets ===")
    
    all_action_changes = []
    
    for split_name, dataset in datasets.items():
        print(f"Extracting from {split_name} dataset ({len(dataset)} samples)...")
        
        for i in range(len(dataset)):
            try:
                # Get data from dataset (image, action_change, at1, at2)
                _, action_change_6dof, _, _ = dataset[i]
                
                # Convert tensor to numpy if needed
                if hasattr(action_change_6dof, 'numpy'):
                    action_change = action_change_6dof.numpy()
                else:
                    action_change = action_change_6dof
                
                if len(action_change) == 6:  # Ensure it's 6DOF
                    all_action_changes.append(action_change)
                    
            except Exception as e:
                print(f"Warning: Error extracting sample {i} from {split_name}: {e}")
                continue
    
    action_changes_array = np.array(all_action_changes)
    print(f"Successfully extracted {len(action_changes_array)} action_change_6dof samples")
    print(f"Data shape: {action_changes_array.shape}")
    
    return action_changes_array


def extract_all_actions_from_datasets(datasets: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract all action data (at1, action_change, at2) from datasets"""
    print("\n=== Extracting All Action Data from Datasets ===")
    
    all_at1_actions = []
    all_action_changes = []
    all_at2_actions = []
    
    for split_name, dataset in datasets.items():
        print(f"Extracting from {split_name} dataset ({len(dataset)} samples)...")
        
        for i in range(len(dataset)):
            try:
                # Get data from dataset (image, action_change, at1, at2)
                _, action_change_6dof, at1_6dof, at2_6dof = dataset[i]
                
                # Convert tensors to numpy if needed
                def to_numpy(tensor):
                    if hasattr(tensor, 'numpy'):
                        return tensor.numpy()
                    return tensor
                
                action_change = to_numpy(action_change_6dof)
                at1_action = to_numpy(at1_6dof)
                at2_action = to_numpy(at2_6dof)
                
                if len(action_change) == 6 and len(at1_action) == 6 and len(at2_action) == 6:
                    all_action_changes.append(action_change)
                    all_at1_actions.append(at1_action)
                    all_at2_actions.append(at2_action)
                    
            except Exception as e:
                print(f"Warning: Error extracting sample {i} from {split_name}: {e}")
                continue
    
    at1_array = np.array(all_at1_actions)
    action_changes_array = np.array(all_action_changes)
    at2_array = np.array(all_at2_actions)
    
    print(f"Successfully extracted:")
    print(f"  AT1 actions: {at1_array.shape}")
    print(f"  Action changes: {action_changes_array.shape}")
    print(f"  AT2 actions: {at2_array.shape}")
    
    return at1_array, action_changes_array, at2_array


def create_distribution_plots(action_changes: np.ndarray, output_dir: str):
    """Create distribution plots for action_change_6dof"""
    print("\n=== Creating Distribution Plots ===")
    
    # Axis names
    axis_names = ['X (mm)', 'Y (mm)', 'Z (mm)', 'Roll (deg)', 'Pitch (deg)', 'Yaw (deg)']
    axis_short = ['X', 'Y', 'Z', 'Roll', 'Pitch', 'Yaw']
    
    # Set plot style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # 1. Overall distribution (2x3 subplots)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i in range(6):
        data = action_changes[:, i]
        
        # Histogram + density curve
        axes[i].hist(data, bins=50, alpha=0.7, density=True, color=f'C{i}')
        
        # Add statistical lines
        mean_val = np.mean(data)
        std_val = np.std(data)
        axes[i].axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.3f}')
        axes[i].axvline(mean_val + std_val, color='orange', linestyle=':', alpha=0.7, label=f'+1σ: {mean_val + std_val:.3f}')
        axes[i].axvline(mean_val - std_val, color='orange', linestyle=':', alpha=0.7, label=f'-1σ: {mean_val - std_val:.3f}')
        
        axes[i].set_title(f'{axis_names[i]} Distribution')
        axes[i].set_xlabel('Value')
        axes[i].set_ylabel('Density')
        axes[i].legend(fontsize=8)
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'action_change_distributions.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Box plot - Fix matplotlib deprecation warning
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    box_data = [action_changes[:, i] for i in range(6)]
    bp = ax.boxplot(box_data, tick_labels=axis_short, patch_artist=True)  # Changed from labels to tick_labels
    
    # Set colors
    colors = plt.cm.Set3(np.linspace(0, 1, 6))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_title('Action Change 6DOF Box Plot', fontsize=16)
    ax.set_ylabel('Value', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'action_change_boxplot.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Correlation heatmap
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    correlation_matrix = np.corrcoef(action_changes.T)
    
    sns.heatmap(correlation_matrix, 
                annot=True, 
                cmap='coolwarm', 
                center=0,
                xticklabels=axis_short,
                yticklabels=axis_short,
                ax=ax)
    
    ax.set_title('Action Change 6DOF Correlation Matrix', fontsize=16)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'action_change_correlation.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Statistical summary table
    stats_df = pd.DataFrame({
        'Axis': axis_names,
        'Mean': [np.mean(action_changes[:, i]) for i in range(6)],
        'Std': [np.std(action_changes[:, i]) for i in range(6)],
        'Min': [np.min(action_changes[:, i]) for i in range(6)],
        'Max': [np.max(action_changes[:, i]) for i in range(6)],
        'Median': [np.median(action_changes[:, i]) for i in range(6)],
        'Q25': [np.percentile(action_changes[:, i], 25) for i in range(6)],
        'Q75': [np.percentile(action_changes[:, i], 75) for i in range(6)]
    })
    
    # Save statistics table
    stats_file = os.path.join(output_dir, 'action_change_statistics.csv')
    stats_df.to_csv(stats_file, index=False)
    
    print(f"Distribution plots saved to: {output_dir}")
    print(f"Statistics table saved to: {stats_file}")
    
    return stats_df


def create_comprehensive_action_analysis(at1_actions: np.ndarray, action_changes: np.ndarray, at2_actions: np.ndarray, output_dir: str):
    """Create comprehensive analysis of all action types"""
    print("\n=== Creating Comprehensive Action Analysis ===")
    
    axis_names = ['X (mm)', 'Y (mm)', 'Z (mm)', 'Roll (deg)', 'Pitch (deg)', 'Yaw (deg)']
    axis_short = ['X', 'Y', 'Z', 'Roll', 'Pitch', 'Yaw']
    
    # Create comprehensive comparison plot
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    
    # Plot distributions for each axis
    for i in range(6):
        row, col = i // 2, i % 2
        
        # Plot all three action types
        axes[row, col].hist(at1_actions[:, i], bins=30, alpha=0.5, label='AT1 (Initial)', color='#FF6B6B', density=True)
        axes[row, col].hist(action_changes[:, i], bins=30, alpha=0.5, label='Action Change', color='#4ECDC4', density=True)
        axes[row, col].hist(at2_actions[:, i], bins=30, alpha=0.5, label='AT2 (Final)', color='#45B7D1', density=True)
        
        axes[row, col].set_title(f'{axis_names[i]} Distribution Comparison')
        axes[row, col].set_xlabel('Value')
        axes[row, col].set_ylabel('Density')
        axes[row, col].legend()
        axes[row, col].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comprehensive_action_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create statistics comparison table
    stats_comparison = []
    
    for i, axis_name in enumerate(axis_names):
        stats_comparison.append({
            'Axis': axis_name,
            'AT1_Mean': np.mean(at1_actions[:, i]),
            'AT1_Std': np.std(at1_actions[:, i]),
            'AT1_Range': f"[{np.min(at1_actions[:, i]):.3f}, {np.max(at1_actions[:, i]):.3f}]",
            'Change_Mean': np.mean(action_changes[:, i]),
            'Change_Std': np.std(action_changes[:, i]),
            'Change_Range': f"[{np.min(action_changes[:, i]):.3f}, {np.max(action_changes[:, i]):.3f}]",
            'AT2_Mean': np.mean(at2_actions[:, i]),
            'AT2_Std': np.std(at2_actions[:, i]),
            'AT2_Range': f"[{np.min(at2_actions[:, i]):.3f}, {np.max(at2_actions[:, i]):.3f}]"
        })
    
    stats_df = pd.DataFrame(stats_comparison)
    stats_file = os.path.join(output_dir, 'comprehensive_action_statistics.csv')
    stats_df.to_csv(stats_file, index=False)
    
    print(f"Comprehensive action analysis saved to: {output_dir}")
    
    return stats_df


def create_patient_distribution_plot(splits_info: Dict, output_dir: str):
    """Create patient distribution plots"""
    print("\n=== Creating Patient Distribution Plots ===")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Patient count distribution (pie chart)
    patient_counts = [splits_info[split]['patient_count'] for split in ['train', 'val', 'test']]
    labels = ['Train', 'Validation', 'Test']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    axes[0, 0].pie(patient_counts, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    axes[0, 0].set_title('Patient Count Distribution')
    
    # 2. Sample count distribution (pie chart)
    sample_counts = [splits_info[split]['total_samples'] for split in ['train', 'val', 'test']]
    
    axes[0, 1].pie(sample_counts, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    axes[0, 1].set_title('Sample Count Distribution')
    
    # 3. Samples per patient (bar chart)
    all_patients = []
    all_counts = []
    all_splits = []
    
    for split_name, info in splits_info.items():
        for patient, count in info['sample_counts'].items():
            all_patients.append(patient)
            all_counts.append(count)
            all_splits.append(split_name)
    
    # Create DataFrame for plotting
    df = pd.DataFrame({
        'Patient': all_patients,
        'Sample_Count': all_counts,
        'Split': all_splits
    })
    
    # Sort by sample count
    df = df.sort_values('Sample_Count', ascending=True)
    
    # Set color mapping
    split_colors = {'train': '#FF6B6B', 'val': '#4ECDC4', 'test': '#45B7D1'}
    colors_mapped = [split_colors[split] for split in df['Split']]
    
    axes[1, 0].barh(df['Patient'], df['Sample_Count'], color=colors_mapped)
    axes[1, 0].set_xlabel('Sample Count')
    axes[1, 0].set_ylabel('Patient ID')
    axes[1, 0].set_title('Samples per Patient')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=split_colors[split], label=split.capitalize()) 
                      for split in ['train', 'val', 'test']]
    axes[1, 0].legend(handles=legend_elements)
    
    # 4. Statistical summary
    summary_text = f"""Dataset Statistical Summary:

Total Patients: {sum(splits_info[split]['patient_count'] for split in ['train', 'val', 'test'])}
Total Samples: {sum(splits_info[split]['total_samples'] for split in ['train', 'val', 'test'])}

Train Set: {splits_info['train']['patient_count']} patients, {splits_info['train']['total_samples']} samples
Val Set: {splits_info['val']['patient_count']} patients, {splits_info['val']['total_samples']} samples
Test Set: {splits_info['test']['patient_count']} patients, {splits_info['test']['total_samples']} samples

Average Samples per Patient:
Train: {splits_info['train']['total_samples'] / max(1, splits_info['train']['patient_count']):.1f}
Val: {splits_info['val']['total_samples'] / max(1, splits_info['val']['patient_count']):.1f}
Test: {splits_info['test']['total_samples'] / max(1, splits_info['test']['patient_count']):.1f}"""
    
    axes[1, 1].text(0.1, 0.5, summary_text, fontsize=11, 
                    verticalalignment='center', fontfamily='monospace')
    axes[1, 1].set_title('Statistical Summary')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'patient_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Patient distribution plots saved to: {output_dir}")


def save_detailed_report(splits_info: Dict, stats_df: pd.DataFrame, comprehensive_stats_df: pd.DataFrame, output_dir: str):
    """Save detailed report"""
    report_file = os.path.join(output_dir, 'dataset_analysis_report.txt')
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("Cardiac Dreamer Dataset Analysis Report\n")
        f.write("(Using CrossPatientTransitionsDataset - Same as Training)\n")
        f.write("=" * 80 + "\n\n")
        
        # Patient grouping information
        f.write("1. Patient Grouping Information\n")
        f.write("-" * 40 + "\n")
        
        total_patients = sum(splits_info[split]['patient_count'] for split in ['train', 'val', 'test'])
        total_samples = sum(splits_info[split]['total_samples'] for split in ['train', 'val', 'test'])
        
        f.write(f"Total: {total_patients} patients, {total_samples} samples\n\n")
        
        for split_name, info in splits_info.items():
            f.write(f"{split_name.upper()} Set:\n")
            f.write(f"  Patient Count: {info['patient_count']} ({info['patient_count']/total_patients:.1%})\n")
            f.write(f"  Total Samples: {info['total_samples']} ({info['total_samples']/total_samples:.1%})\n")
            f.write(f"  Patient List: {', '.join(info['patients'])}\n")
            f.write(f"  Samples per Patient:\n")
            for patient, count in info['sample_counts'].items():
                f.write(f"    {patient}: {count} samples\n")
            f.write("\n")
        
        # Action Change Statistics
        f.write("2. Action Change 6DOF Statistics\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total Samples: {len(stats_df)} axes x {total_samples} samples\n\n")
        
        for _, row in stats_df.iterrows():
            f.write(f"{row['Axis']}:\n")
            f.write(f"  Mean: {row['Mean']:.4f}\n")
            f.write(f"  Std Dev: {row['Std']:.4f}\n")
            f.write(f"  Range: [{row['Min']:.4f}, {row['Max']:.4f}]\n")
            f.write(f"  Median: {row['Median']:.4f}\n")
            f.write(f"  IQR: [{row['Q25']:.4f}, {row['Q75']:.4f}]\n\n")
        
        # Comprehensive Action Analysis
        f.write("3. Comprehensive Action Analysis (AT1, Change, AT2)\n")
        f.write("-" * 40 + "\n")
        
        for _, row in comprehensive_stats_df.iterrows():
            f.write(f"{row['Axis']}:\n")
            f.write(f"  AT1 (Initial): Mean={row['AT1_Mean']:.4f}, Std={row['AT1_Std']:.4f}, Range={row['AT1_Range']}\n")
            f.write(f"  Change: Mean={row['Change_Mean']:.4f}, Std={row['Change_Std']:.4f}, Range={row['Change_Range']}\n")
            f.write(f"  AT2 (Final): Mean={row['AT2_Mean']:.4f}, Std={row['AT2_Std']:.4f}, Range={row['AT2_Range']}\n\n")
        
        f.write("4. Data Loading Method\n")
        f.write("-" * 40 + "\n")
        f.write("This analysis uses the same CrossPatientTransitionsDataset class as training,\n")
        f.write("ensuring complete consistency with the actual training data pipeline.\n\n")
        
        f.write("5. Generated Files\n")
        f.write("-" * 40 + "\n")
        f.write("  - patient_distribution.png: Patient distribution plots\n")
        f.write("  - action_change_distributions.png: Action change distribution plots\n")
        f.write("  - action_change_boxplot.png: Action change box plot\n")
        f.write("  - action_change_correlation.png: Action change correlation matrix\n")
        f.write("  - comprehensive_action_analysis.png: All action types comparison\n")
        f.write("  - action_change_statistics.csv: Action change statistics table\n")
        f.write("  - comprehensive_action_statistics.csv: All action types statistics\n")
        f.write("  - dataset_analysis_report.txt: This report\n")
    
    print(f"Detailed report saved to: {report_file}")


def main():
    parser = argparse.ArgumentParser(description="Analyze Cardiac Dreamer Dataset using Training Pipeline")
    parser.add_argument("--data_dir", type=str, default="data/processed", 
                       help="Data directory path")
    parser.add_argument("--output_dir", type=str, default="dataset_analysis", 
                       help="Output directory path")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Starting dataset analysis using CrossPatientTransitionsDataset...")
    print(f"Data Directory: {args.data_dir}")
    print(f"Output Directory: {args.output_dir}")
    print("This analysis uses the SAME data loading method as training for consistency.")
    
    # 1. Analyze patient splits using the same method as training
    splits_info, datasets = analyze_patient_splits(args.data_dir)
    
    # 2. Extract all action data from datasets
    at1_actions, action_changes, at2_actions = extract_all_actions_from_datasets(datasets)
    
    # 3. Create distribution plots for action changes
    stats_df = create_distribution_plots(action_changes, args.output_dir)
    
    # 4. Create comprehensive action analysis
    comprehensive_stats_df = create_comprehensive_action_analysis(at1_actions, action_changes, at2_actions, args.output_dir)
    
    # 5. Create patient distribution plots
    create_patient_distribution_plot(splits_info, args.output_dir)
    
    # 6. Save detailed report
    save_detailed_report(splits_info, stats_df, comprehensive_stats_df, args.output_dir)
    
    print(f"\n[SUCCESS] Analysis Complete! All results saved to: {args.output_dir}")  # Changed Unicode symbol
    print("\nGenerated Files:")
    for file in os.listdir(args.output_dir):
        print(f"  - {file}")
    
    print(f"\n[STATS] Key Statistics:")  # Changed Unicode symbol
    print(f"  Total Samples: {len(action_changes)}")
    print(f"  Train: {splits_info['train']['total_samples']} samples from {splits_info['train']['patient_count']} patients")
    print(f"  Val: {splits_info['val']['total_samples']} samples from {splits_info['val']['patient_count']} patients")
    print(f"  Test: {splits_info['test']['total_samples']} samples from {splits_info['test']['patient_count']} patients")


if __name__ == "__main__":
    main() 