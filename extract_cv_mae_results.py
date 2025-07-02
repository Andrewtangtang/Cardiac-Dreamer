#!/usr/bin/env python
"""
Extract MAE Results from Cross-Validation for Patient Comparison

This script evaluates each fold in cross-validation to generate detailed MAE metrics
for each patient group, which can be compared with RSSM model results.

FIXED: Each fold now uses its own training set normalization statistics.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import torch
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import tempfile
import shutil

# Add project root to path
current_file_dir = os.path.dirname(os.path.abspath(__file__))
if current_file_dir not in sys.path:
    sys.path.insert(0, current_file_dir)

from model_evaluation import load_model_from_checkpoint, generate_predictions, calculate_metrics

def get_patient_groups():
    """Define the 5 patient groups for cross-validation"""
    return {
        'patient1': ['data_0513_01', 'data_0513_02', 'data_0513_03', 'data_0513_04', 'data_0513_05'],
        'patient2': ['data_0513_06', 'data_0513_07', 'data_0513_08', 'data_0513_09'],
        'patient3': ['data_0513_11', 'data_0513_12', 'data_0513_13', 'data_0513_14'],
        'patient4': ['data_0513_16', 'data_0513_17', 'data_0513_18', 'data_0513_19', 'data_0513_20','data_0513_21'],
        'patient5': ['data_0513_22', 'data_0513_23', 'data_0513_24', 'data_0513_25', 'data_0513_26']
    }

def create_fold_specific_datasets(data_dir, val_patient_group, fold_idx):
    """
    Create datasets with fold-specific normalization statistics
    This ensures each fold uses its own training set statistics for normalization
    """
    import torchvision.transforms as transforms
    from src.data import CrossPatientTransitionsDataset
    
    patient_groups = get_patient_groups()
    
    # Get validation patients for this fold
    val_patients = patient_groups[val_patient_group]
    
    # Get training patients (all other groups)
    train_patients = []
    for group_name, patients in patient_groups.items():
        if group_name != val_patient_group:
            train_patients.extend(patients)
    
    print(f"   Creating fold-specific normalization for fold {fold_idx}")
    print(f"   Training patients: {train_patients}")
    print(f"   Validation patients: {val_patients}")
    
    # Create temporary directory for this fold's normalization stats
    fold_temp_dir = tempfile.mkdtemp(prefix=f"fold_{fold_idx}_")
    fold_data_dir = os.path.join(fold_temp_dir, "processed")
    
    try:
        # Copy patient data to temporary directory
        for patient_id in (train_patients + val_patients):
            src_patient_dir = os.path.join(data_dir, patient_id)
            dst_patient_dir = os.path.join(fold_data_dir, patient_id)
            if os.path.exists(src_patient_dir):
                shutil.copytree(src_patient_dir, dst_patient_dir)
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        # Step 1: Create training dataset to compute fold-specific normalization stats
        print(f"   Computing normalization statistics for fold {fold_idx}...")
        train_dataset = CrossPatientTransitionsDataset(
            data_dir=fold_data_dir,
            transform=transform,
            split="train",
            train_patients=train_patients,
            val_patients=val_patients,
            test_patients=[],
            small_subset=False,
            normalize_actions=True  # This will compute and save stats for this fold
        )
        
        # Step 2: Create validation dataset using the same stats
        print(f"   Creating validation dataset for fold {fold_idx}...")
        val_dataset = CrossPatientTransitionsDataset(
            data_dir=fold_data_dir,
            transform=transform,
            split="val",
            train_patients=train_patients,
            val_patients=val_patients,
            test_patients=[],
            small_subset=False,
            normalize_actions=True  # This will load the stats computed above
        )
        
        print(f"   Fold {fold_idx} normalization stats:")
        print(f"   Action mean: {val_dataset.action_mean}")
        print(f"   Action std: {val_dataset.action_std}")
        
        return val_dataset, fold_temp_dir
        
    except Exception as e:
        # Clean up on error
        if os.path.exists(fold_temp_dir):
            shutil.rmtree(fold_temp_dir)
        raise e

def extract_mae_from_cv_results(cv_output_dir, data_dir="data/processed"):
    """
    Extract MAE for each patient group from cross-validation results
    Now with correct fold-specific normalization
    """
    print(f"Analyzing cross-validation results: {cv_output_dir}")
    
    # Load cross-validation summary
    summary_file = os.path.join(cv_output_dir, "cv_summary_report.json")
    if not os.path.exists(summary_file):
        raise FileNotFoundError(f"Cross-validation summary not found: {summary_file}")
    
    with open(summary_file, 'r') as f:
        cv_summary = json.load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Store MAE results for each patient group
    mae_results = {}
    temp_dirs_to_cleanup = []
    
    print(f"\nStarting MAE evaluation for each patient group...")
    
    try:
        for fold_detail in cv_summary['fold_details']:
            fold_idx = fold_detail['fold_idx']
            val_patient_group = fold_detail['val_patient_group']
            best_checkpoint = fold_detail['best_model_path']
            
            print(f"\nFold {fold_idx}: Evaluating {val_patient_group}")
            print(f"   Checkpoint: {os.path.basename(best_checkpoint)}")
            
            try:
                # Load best model
                model = load_model_from_checkpoint(best_checkpoint)
                model = model.to(device)
                
                # Create fold-specific validation dataset with correct normalization
                val_dataset, fold_temp_dir = create_fold_specific_datasets(
                    data_dir, val_patient_group, fold_idx
                )
                temp_dirs_to_cleanup.append(fold_temp_dir)
                
                if len(val_dataset) == 0:
                    print(f"   Warning: No validation data for {val_patient_group}")
                    continue
                
                # Create data loader
                from torch.utils.data import DataLoader
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=16,
                    shuffle=False,
                    num_workers=2,
                    pin_memory=False
                )
                
                print(f"   Generating predictions... (samples: {len(val_dataset)})")
                
                # Generate predictions and calculate MAE
                predictions, ground_truth = generate_predictions(model, val_loader, device)
                
                # Get fold-specific normalization stats for denormalization
                fold_action_mean = val_dataset.action_mean
                fold_action_std = val_dataset.action_std
                
                print(f"   Using fold-specific normalization stats for denormalization:")
                print(f"     Action mean: {fold_action_mean}")
                print(f"     Action std: {fold_action_std}")
                
                # Calculate metrics with denormalization to original physical units
                metrics = calculate_metrics(predictions, ground_truth, fold_action_mean, fold_action_std)
                
                # Extract MAE data
                mae_data = {}
                for dim in ['X', 'Y', 'Z', 'Roll', 'Pitch', 'Yaw']:
                    mae_data[dim] = {
                        'mae': metrics[dim]['mae'],
                        'unit': metrics[dim]['unit'],
                        'r2_score': metrics[dim]['r2_score'],
                        'correlation': metrics[dim]['correlation']
                    }
                
                # Calculate average MAE for 3D translation and rotation
                translation_mae = np.mean([metrics['X']['mae'], metrics['Y']['mae'], metrics['Z']['mae']])
                rotation_mae = np.mean([metrics['Roll']['mae'], metrics['Pitch']['mae'], metrics['Yaw']['mae']])
                
                # Calculate average R2 and correlation
                avg_r2 = np.mean([metrics[dim]['r2_score'] for dim in ['X', 'Y', 'Z', 'Roll', 'Pitch', 'Yaw']])
                avg_correlation = np.mean([metrics[dim]['correlation'] for dim in ['X', 'Y', 'Z', 'Roll', 'Pitch', 'Yaw']])
                
                patient_groups = get_patient_groups()
                mae_results[val_patient_group] = {
                    'fold': fold_idx,
                    'patients': patient_groups[val_patient_group],
                    'sample_count': len(val_dataset),
                    'mae_by_dimension': mae_data,
                    'translation_mae_mm': translation_mae,
                    'rotation_mae_deg': rotation_mae,
                    'validation_loss': fold_detail['best_val_loss'],
                    'avg_r2_score': avg_r2,
                    'avg_correlation': avg_correlation,
                    'normalization_stats': {
                        'action_mean': val_dataset.action_mean.tolist(),
                        'action_std': val_dataset.action_std.tolist()
                    }
                }
                
                print(f"   Completed! CORRECTED 3D Translation MAE: {translation_mae:.3f}mm, Rotation MAE: {rotation_mae:.3f}deg")
                print(f"   Average R2: {avg_r2:.3f}, Average Correlation: {avg_correlation:.3f}")
                print(f"   NOTE: MAE values are now in original physical units (mm/degrees)")
                
            except Exception as e:
                print(f"   Error: {e}")
                continue
    
    finally:
        # Clean up temporary directories
        for temp_dir in temp_dirs_to_cleanup:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                print(f"   Cleaned up temporary directory: {temp_dir}")
    
    return mae_results

def generate_comparison_table(mae_results, output_dir):
    """Generate table for comparison with RSSM"""
    print(f"\nGenerating MAE comparison table...")
    
    # Create table data
    table_data = []
    
    # Sort results by fold order
    sorted_results = sorted(mae_results.items(), key=lambda x: x[1]['fold'])
    
    for group_name, data in sorted_results:
        # Format patient list
        patients_str = ', '.join(data['patients'])
        
        # Extract dimension-wise MAE
        x_mae = data['mae_by_dimension']['X']['mae']
        y_mae = data['mae_by_dimension']['Y']['mae'] 
        z_mae = data['mae_by_dimension']['Z']['mae']
        roll_mae = data['mae_by_dimension']['Roll']['mae']
        pitch_mae = data['mae_by_dimension']['Pitch']['mae']
        yaw_mae = data['mae_by_dimension']['Yaw']['mae']
        
        # Extract R2 scores
        x_r2 = data['mae_by_dimension']['X']['r2_score']
        y_r2 = data['mae_by_dimension']['Y']['r2_score']
        z_r2 = data['mae_by_dimension']['Z']['r2_score']
        
        table_data.append({
            'Fold': f"P{data['fold']}",
            'Patient_Group': group_name,
            'Patients': patients_str,
            'Sample_Count': data['sample_count'],
            'X_MAE_mm': round(x_mae, 3),
            'Y_MAE_mm': round(y_mae, 3),
            'Z_MAE_mm': round(z_mae, 3),
            'Translation_MAE_mm': round(data['translation_mae_mm'], 3),
            'Roll_MAE_deg': round(roll_mae, 3),
            'Pitch_MAE_deg': round(pitch_mae, 3),
            'Yaw_MAE_deg': round(yaw_mae, 3),
            'Rotation_MAE_deg': round(data['rotation_mae_deg'], 3),
            'Validation_Loss': round(data['validation_loss'], 6),
            'X_R2': round(x_r2, 4),
            'Y_R2': round(y_r2, 4),
            'Z_R2': round(z_r2, 4),
            'Avg_R2': round(data['avg_r2_score'], 4),
            'Avg_Correlation': round(data['avg_correlation'], 4)
        })
    
    # Create DataFrame
    df = pd.DataFrame(table_data)
    
    # Save detailed table
    detailed_file = os.path.join(output_dir, "detailed_mae_comparison_corrected.csv")
    df.to_csv(detailed_file, index=False)
    
    # Create simplified comparison table (similar to your format)
    simple_table = []
    for _, row in df.iterrows():
        simple_table.append({
            'Fold': row['Fold'],
            'Dreamer_Translation_MAE': f"{row['Translation_MAE_mm']:.1f}",
            'Dreamer_Rotation_MAE': f"{row['Rotation_MAE_deg']:.1f}",
            'RSSM_Translation_MAE': "To_be_filled",  # You need to manually fill RSSM results
            'RSSM_Rotation_MAE': "To_be_filled",
            'Translation_Improvement': "To_be_calculated",
            'Rotation_Improvement': "To_be_calculated",
            'Avg_R2_Score': f"{row['Avg_R2']:.3f}",
            'Avg_Correlation': f"{row['Avg_Correlation']:.3f}"
        })
    
    simple_df = pd.DataFrame(simple_table)
    simple_file = os.path.join(output_dir, "simple_mae_comparison_corrected.csv")
    simple_df.to_csv(simple_file, index=False)
    
    # Print results
    print(f"\nCORRECTED MAE Comparison Results:")
    print("=" * 120)
    print(df[['Fold', 'Patient_Group', 'Translation_MAE_mm', 'Rotation_MAE_deg', 'Avg_R2', 'Avg_Correlation', 'Sample_Count']].to_string(index=False))
    
    # Calculate statistics
    print(f"\nStatistical Summary (CORRECTED):")
    print(f"   Mean 3D Translation MAE: {df['Translation_MAE_mm'].mean():.3f} ± {df['Translation_MAE_mm'].std():.3f} mm")
    print(f"   Mean Rotation MAE: {df['Rotation_MAE_deg'].mean():.3f} ± {df['Rotation_MAE_deg'].std():.3f} deg")
    print(f"   Mean R2 Score: {df['Avg_R2'].mean():.3f} ± {df['Avg_R2'].std():.3f}")
    print(f"   Mean Correlation: {df['Avg_Correlation'].mean():.3f} ± {df['Avg_Correlation'].std():.3f}")
    print(f"   Best 3D Translation MAE: {df['Translation_MAE_mm'].min():.3f} mm ({df.loc[df['Translation_MAE_mm'].idxmin(), 'Patient_Group']})")
    print(f"   Best Rotation MAE: {df['Rotation_MAE_deg'].min():.3f} deg ({df.loc[df['Rotation_MAE_deg'].idxmin(), 'Patient_Group']})")
    print(f"   Best R2 Score: {df['Avg_R2'].max():.3f} ({df.loc[df['Avg_R2'].idxmax(), 'Patient_Group']})")
    
    print(f"\nResults saved (CORRECTED):")
    print(f"   Detailed table: {detailed_file}")
    print(f"   Simple table: {simple_file}")
    
    return df

def create_mae_visualization(mae_results, df, output_dir):
    """Create MAE visualization plots with corrected data"""
    print(f"\nCreating CORRECTED MAE visualization plots...")
    
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Cross-Validation MAE Results by Patient Group (CORRECTED NORMALIZATION)', fontsize=16, fontweight='bold')
    
    # Color configuration
    colors = plt.cm.Set3(np.linspace(0, 1, len(df)))
    
    # 1. 3D Translation MAE
    ax1 = axes[0, 0]
    bars1 = ax1.bar(df['Fold'], df['Translation_MAE_mm'], color=colors, alpha=0.8, edgecolor='black')
    ax1.set_title('3D Translation MAE by Patient Group')
    ax1.set_xlabel('Patient Group')
    ax1.set_ylabel('MAE (mm)')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars1, df['Translation_MAE_mm']):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{val:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Rotation MAE
    ax2 = axes[0, 1]
    bars2 = ax2.bar(df['Fold'], df['Rotation_MAE_deg'], color=colors, alpha=0.8, edgecolor='black')
    ax2.set_title('Rotation MAE by Patient Group')
    ax2.set_xlabel('Patient Group')
    ax2.set_ylabel('MAE (degrees)')
    ax2.grid(True, alpha=0.3)
    
    for bar, val in zip(bars2, df['Rotation_MAE_deg']):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{val:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # 3. R2 Score comparison
    ax3 = axes[0, 2]
    bars3 = ax3.bar(df['Fold'], df['Avg_R2'], color=colors, alpha=0.8, edgecolor='black')
    ax3.set_title('Average R² Score by Patient Group')
    ax3.set_xlabel('Patient Group')
    ax3.set_ylabel('R² Score')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='Zero Line')
    ax3.legend()
    
    for bar, val in zip(bars3, df['Avg_R2']):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 4. Dimension-wise MAE comparison
    ax4 = axes[1, 0]
    dimensions = ['X', 'Y', 'Z', 'Roll', 'Pitch', 'Yaw']
    
    # Plot lines for each patient group
    for i, (group_name, data) in enumerate(mae_results.items()):
        values = [
            data['mae_by_dimension']['X']['mae'],
            data['mae_by_dimension']['Y']['mae'], 
            data['mae_by_dimension']['Z']['mae'],
            data['mae_by_dimension']['Roll']['mae'],
            data['mae_by_dimension']['Pitch']['mae'],
            data['mae_by_dimension']['Yaw']['mae']
        ]
        ax4.plot(dimensions, values, marker='o', linewidth=2, label=group_name, color=colors[i])
    
    ax4.set_title('MAE by Dimension for Each Patient Group')
    ax4.set_ylabel('MAE')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.tick_params(axis='x', rotation=45)
    
    # 5. Correlation comparison
    ax5 = axes[1, 1]
    bars5 = ax5.bar(df['Fold'], df['Avg_Correlation'], color=colors, alpha=0.8, edgecolor='black')
    ax5.set_title('Average Correlation by Patient Group')
    ax5.set_xlabel('Patient Group')
    ax5.set_ylabel('Correlation')
    ax5.grid(True, alpha=0.3)
    ax5.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='Zero Line')
    ax5.legend()
    
    for bar, val in zip(bars5, df['Avg_Correlation']):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 6. Sample count distribution
    ax6 = axes[1, 2]
    bars6 = ax6.bar(df['Fold'], df['Sample_Count'], color=colors, alpha=0.8, edgecolor='black')
    ax6.set_title('Sample Count by Patient Group')
    ax6.set_xlabel('Patient Group')
    ax6.set_ylabel('Number of Samples')
    ax6.grid(True, alpha=0.3)
    
    for bar, val in zip(bars6, df['Sample_Count']):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{val}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    # Save plot
    plot_file = os.path.join(output_dir, "mae_visualization_corrected.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   CORRECTED visualization saved: {plot_file}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Extract cross-validation MAE results for patient comparison (CORRECTED)")
    parser.add_argument("cv_output_dir", type=str, help="Cross-validation output directory")
    parser.add_argument("--data_dir", type=str, default="data/processed", help="Data directory")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory (default: cv_dir/mae_analysis_corrected)")
    
    args = parser.parse_args()
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(args.cv_output_dir, "mae_analysis_corrected")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        print("Starting CORRECTED cross-validation MAE extraction...")
        print("FIXED: Each fold now uses its own training set normalization statistics")
        print(f"CV directory: {args.cv_output_dir}")
        print(f"Data directory: {args.data_dir}")
        print(f"Output directory: {args.output_dir}")
        
        # 1. Extract MAE results with corrected normalization
        mae_results = extract_mae_from_cv_results(args.cv_output_dir, args.data_dir)
        
        if not mae_results:
            print("No valid MAE results found")
            return
        
        # 2. Generate comparison table
        df = generate_comparison_table(mae_results, args.output_dir)
        
        # 3. Create visualization
        create_mae_visualization(mae_results, df, args.output_dir)
        
        # 4. Save raw results
        results_file = os.path.join(args.output_dir, "raw_mae_results_corrected.json")
        with open(results_file, 'w') as f:
            json.dump(mae_results, f, indent=2)
        
        print(f"\nCORRECTED MAE analysis completed!")
        print(f"Key improvements:")
        print(f"  - Each fold uses its own training set normalization statistics")
        print(f"  - Added R2 scores and correlation metrics")
        print(f"  - Improved error handling and cleanup")
        print(f"\nYou can now compare results with RSSM model")
        print(f"Files with '_corrected' suffix contain the accurate results")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 