#!/usr/bin/env python
"""
Cross-Validation Results Analyzer for Cardiac Dreamer

This script analyzes completed cross-validation results and provides
detailed insights and comparisons.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path
from datetime import datetime

def load_cv_results(cv_output_dir):
    """Load cross-validation results from output directory"""
    
    # Check if results exist
    summary_file = os.path.join(cv_output_dir, "cv_summary_report.json")
    csv_file = os.path.join(cv_output_dir, "cv_detailed_results.csv")
    
    if not os.path.exists(summary_file):
        raise FileNotFoundError(f"Summary report not found: {summary_file}")
    
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"Detailed results not found: {csv_file}")
    
    # Load JSON summary
    with open(summary_file, 'r') as f:
        summary = json.load(f)
    
    # Load CSV details
    detailed_df = pd.read_csv(csv_file)
    
    return summary, detailed_df

def print_detailed_analysis(summary, detailed_df):
    """Print detailed analysis of cross-validation results"""
    
    print("=" * 100)
    print("CROSS-VALIDATION RESULTS ANALYSIS")
    print("=" * 100)
    
    # Basic statistics
    cv_stats = summary['cross_validation_summary']
    loss_stats = cv_stats['validation_loss_statistics']
    
    print(f"\n BASIC STATISTICS:")
    print(f"   Number of Folds: {cv_stats['total_folds']}")
    print(f"   Epochs per Fold: {cv_stats['epochs_per_fold']}")
    print(f"   Mean Validation Loss: {loss_stats['mean']:.6f}")
    print(f"   Standard Deviation: {loss_stats['std']:.6f}")
    print(f"   Coefficient of Variation: {(loss_stats['std']/loss_stats['mean']*100):.2f}%")
    print(f"   Best Performance: {loss_stats['min']:.6f}")
    print(f"   Worst Performance: {loss_stats['max']:.6f}")
    print(f"   Performance Range: {loss_stats['max'] - loss_stats['min']:.6f}")
    
    # Patient group analysis
    print(f"\n PATIENT GROUP PERFORMANCE RANKING:")
    patient_perf = summary['patient_performance']
    sorted_patients = sorted(patient_perf.items(), key=lambda x: x[1]['validation_loss'])
    
    for i, (group, perf) in enumerate(sorted_patients):
        print(f"   {group}: {perf['validation_loss']:.6f}")
        print(f"      Patients: {', '.join(perf['patients'])}")
        print(f"      Sample Count: {perf['sample_count']}")
        improvement_needed = ((perf['validation_loss'] - loss_stats['min']) / loss_stats['min'] * 100)
        if improvement_needed > 0:
            print(f"      Gap from best: +{improvement_needed:.2f}%")
        print()
    
    # Fold-by-fold analysis
    print(f"\n FOLD-BY-FOLD PERFORMANCE:")
    for detail in summary['fold_details']:
        fold_num = detail['fold_idx']
        val_loss = detail['best_val_loss']
        group = detail['val_patient_group']
        train_samples = detail['train_samples']
        val_samples = detail['val_samples']
        
        # Performance indicator
        if val_loss == loss_stats['min']:
            indicator = "BEST"
        elif val_loss == loss_stats['max']:
            indicator = "WORST"
        elif val_loss < loss_stats['mean']:
            indicator = "ABOVE AVG"
        else:
            indicator = "BELOW AVG"
        
        print(f"   Fold {fold_num}: {val_loss:.6f} {indicator}")
        print(f"      Validation Group: {group}")
        print(f"      Data Split: {train_samples} train / {val_samples} val")
        
    # Statistical significance tests
    print(f"\n VARIABILITY ANALYSIS:")
    losses = loss_stats['all_losses']
    
    # Check if differences are statistically meaningful
    relative_std = loss_stats['std'] / loss_stats['mean']
    if relative_std < 0.05:
        variability_assessment = "Very Low - Highly consistent performance"
    elif relative_std < 0.10:
        variability_assessment = "Low - Good consistency"
    elif relative_std < 0.15:
        variability_assessment = "Moderate - Some variation across folds"
    elif relative_std < 0.20:
        variability_assessment = "High - Significant variation across folds"
    else:
        variability_assessment = "Very High - Large variation, may indicate overfitting to specific patients"
    
    print(f"   Relative Standard Deviation: {relative_std:.4f}")
    print(f"   Assessment: {variability_assessment}")
    
    # Outlier detection
    q1 = np.percentile(losses, 25)
    q3 = np.percentile(losses, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    outliers = [loss for loss in losses if loss < lower_bound or loss > upper_bound]
    if outliers:
        print(f"   Outlier Detection: {len(outliers)} potential outlier(s) found")
        for outlier in outliers:
            fold_info = next(d for d in summary['fold_details'] if d['best_val_loss'] == outlier)
            print(f"      Fold {fold_info['fold_idx']} ({fold_info['val_patient_group']}): {outlier:.6f}")
    else:
        print(f"   Outlier Detection: No outliers detected")

def create_advanced_visualizations(summary, detailed_df, output_dir):
    """Create advanced visualization plots"""
    
    print(f"\n CREATING ADVANCED VISUALIZATIONS...")
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    fig = plt.figure(figsize=(20, 16))
    
    # Create a complex subplot layout
    gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
    
    # Extract data
    folds = detailed_df['Fold'].values
    val_losses = detailed_df['Best_Val_Loss'].values
    patient_groups = detailed_df['Validation_Patient_Group'].values
    train_samples = detailed_df['Train_Samples'].values
    val_samples = detailed_df['Val_Samples'].values
    
    # 1. Main performance plot (top row, span 2 columns)
    ax1 = fig.add_subplot(gs[0, :2])
    bars = ax1.bar(folds, val_losses, color='skyblue', alpha=0.8, edgecolor='navy', linewidth=1.5)
    ax1.axhline(np.mean(val_losses), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(val_losses):.4f}')
    ax1.axhline(np.mean(val_losses) + np.std(val_losses), color='orange', linestyle=':', alpha=0.7, label=f'+1 STD')
    ax1.axhline(np.mean(val_losses) - np.std(val_losses), color='orange', linestyle=':', alpha=0.7, label=f'-1 STD')
    ax1.set_title('Cross-Validation Performance by Fold', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Fold')
    ax1.set_ylabel('Best Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, val in zip(bars, val_losses):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{val:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Patient group comparison (top row, span 2 columns)
    ax2 = fig.add_subplot(gs[0, 2:])
    patient_losses = [detailed_df[detailed_df['Validation_Patient_Group'] == group]['Best_Val_Loss'].iloc[0] 
                      for group in detailed_df['Validation_Patient_Group']]
    colors = plt.cm.Set3(np.linspace(0, 1, len(patient_groups)))
    bars2 = ax2.barh(patient_groups, patient_losses, color=colors, alpha=0.8, edgecolor='black')
    ax2.set_title('Performance by Patient Group', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Best Validation Loss')
    ax2.grid(True, alpha=0.3)
    
    # 3. Sample distribution (second row, left)
    ax3 = fig.add_subplot(gs[1, :2])
    x = np.arange(len(folds))
    width = 0.35
    ax3.bar(x - width/2, train_samples, width, label='Training Samples', color='lightgreen', alpha=0.8)
    ax3.bar(x + width/2, val_samples, width, label='Validation Samples', color='coral', alpha=0.8)
    ax3.set_title('Sample Distribution per Fold', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Fold')
    ax3.set_ylabel('Number of Samples')
    ax3.set_xticks(x)
    ax3.set_xticklabels([f'F{f}' for f in folds])
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Loss distribution histogram (second row, right)
    ax4 = fig.add_subplot(gs[1, 2:])
    ax4.hist(val_losses, bins=max(3, len(val_losses)//2), color='purple', alpha=0.7, edgecolor='black', density=True)
    ax4.axvline(np.mean(val_losses), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(val_losses):.4f}')
    ax4.axvline(np.median(val_losses), color='green', linestyle='-.', linewidth=2, label=f'Median: {np.median(val_losses):.4f}')
    ax4.set_title('Distribution of Validation Losses', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Validation Loss')
    ax4.set_ylabel('Density')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Performance consistency (third row, left)
    ax5 = fig.add_subplot(gs[2, :2])
    consistency_score = 1 - (np.std(val_losses) / np.mean(val_losses))
    ax5.pie([consistency_score, 1-consistency_score], labels=['Consistent', 'Variable'], 
            autopct='%1.1f%%', startangle=90, colors=['lightgreen', 'lightcoral'])
    ax5.set_title(f'Performance Consistency\n(CV = {np.std(val_losses)/np.mean(val_losses)*100:.2f}%)', 
                 fontsize=14, fontweight='bold')
    
    # 6. Ranking comparison (third row, right)
    ax6 = fig.add_subplot(gs[2, 2:])
    ranks = range(1, len(patient_groups) + 1)
    sorted_groups = sorted(zip(patient_groups, val_losses), key=lambda x: x[1])
    sorted_group_names = [group for group, _ in sorted_groups]
    sorted_losses = [loss for _, loss in sorted_groups]
    
    bars6 = ax6.bar(ranks, sorted_losses, color=plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(ranks))), 
                   alpha=0.8, edgecolor='black')
    ax6.set_title('Patient Group Ranking', fontsize=14, fontweight='bold')
    ax6.set_xlabel('Rank (1=Best)')
    ax6.set_ylabel('Validation Loss')
    ax6.set_xticks(ranks)
    ax6.set_xticklabels(sorted_group_names, rotation=45)
    ax6.grid(True, alpha=0.3)
    
    # 7. Cross-validation stability metrics (bottom row)
    ax7 = fig.add_subplot(gs[3, :])
    
    # Create stability metrics
    metrics = {
        'Mean Loss': np.mean(val_losses),
        'Std Dev': np.std(val_losses),
        'Min Loss': np.min(val_losses),
        'Max Loss': np.max(val_losses),
        'Range': np.max(val_losses) - np.min(val_losses),
        'CV (%)': np.std(val_losses)/np.mean(val_losses)*100
    }
    
    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())
    
    # Normalize values for display (except CV which is already in %)
    normalized_values = []
    for i, (name, value) in enumerate(metrics.items()):
        if name == 'CV (%)':
            normalized_values.append(value)
        else:
            normalized_values.append(value * 1000)  # Scale up for visibility
    
    bars7 = ax7.bar(metric_names, normalized_values, 
                   color=['blue', 'orange', 'green', 'red', 'purple', 'brown'], alpha=0.7)
    ax7.set_title('Cross-Validation Stability Metrics', fontsize=14, fontweight='bold')
    ax7.set_ylabel('Value (scaled)')
    ax7.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, value in zip(bars7, metric_values):
        height = bar.get_height()
        if 'Loss' in bar.get_label() or bar.get_label() == 'Range':
            ax7.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{value:.4f}', ha='center', va='bottom', fontsize=10)
        else:
            ax7.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{value:.2f}%', ha='center', va='bottom', fontsize=10)
    
    plt.suptitle('Comprehensive Cross-Validation Analysis Report', fontsize=18, fontweight='bold', y=0.98)
    
    # Save the comprehensive plot
    plot_path = os.path.join(output_dir, "comprehensive_cv_analysis.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   Comprehensive analysis plot saved: {plot_path}")

def generate_recommendations(summary, detailed_df):
    """Generate recommendations based on CV results"""
    
    print(f"\n RECOMMENDATIONS:")
    
    loss_stats = summary['cross_validation_summary']['validation_loss_statistics']
    cv = loss_stats['std'] / loss_stats['mean']
    
    # Performance recommendations
    if loss_stats['mean'] > 0.5:
        print(f"   PERFORMANCE: Loss is relatively high ({loss_stats['mean']:.4f}). Consider:")
        print(f"      - Increasing model capacity or training epochs")
        print(f"      - Adjusting learning rate or regularization")
        print(f"      - Adding more data augmentation")
    elif loss_stats['mean'] < 0.2:
        print(f"   PERFORMANCE: Excellent performance ({loss_stats['mean']:.4f})!")
        print(f"      - Consider deploying this model configuration")
    else:
        print(f"   PERFORMANCE: Good performance ({loss_stats['mean']:.4f})")
        print(f"      - Fine-tuning may yield further improvements")
    
    # Consistency recommendations
    if cv > 0.15:
        print(f"   CONSISTENCY: High variability (CV={cv:.3f}). Consider:")
        print(f"      - Patient-specific fine-tuning approaches")
        print(f"      - Domain adaptation techniques")
        print(f"      - Collecting more data from challenging patients")
    elif cv < 0.05:
        print(f"   CONSISTENCY: Excellent consistency (CV={cv:.3f})!")
        print(f"      - Model generalizes well across patients")
    else:
        print(f"   CONSISTENCY: Good consistency (CV={cv:.3f})")
    
    # Patient-specific recommendations
    patient_perf = summary['patient_performance']
    worst_patient = min(patient_perf.items(), key=lambda x: x[1]['validation_loss'])
    best_patient = max(patient_perf.items(), key=lambda x: x[1]['validation_loss'])
    
    print(f"   PATIENT-SPECIFIC:")
    print(f"      - Best performing: {best_patient[0]} ({best_patient[1]['validation_loss']:.4f})")
    print(f"      - Most challenging: {worst_patient[0]} ({worst_patient[1]['validation_loss']:.4f})")
    
    gap = (worst_patient[1]['validation_loss'] - best_patient[1]['validation_loss']) / best_patient[1]['validation_loss']
    if gap > 0.20:
        print(f"      - Large performance gap ({gap:.1%}) suggests patient-specific challenges")
        print(f"      - Consider analyzing {worst_patient[0]} data for quality issues")

def main():
    """Main analysis function"""
    
    parser = argparse.ArgumentParser(description="Analyze Cross-Validation Results")
    parser.add_argument("cv_output_dir", type=str, help="Cross-validation output directory")
    parser.add_argument("--save_analysis", action="store_true", help="Save detailed analysis to file")
    
    args = parser.parse_args()
    
    try:
        # Load results
        summary, detailed_df = load_cv_results(args.cv_output_dir)
        
        # Print detailed analysis
        print_detailed_analysis(summary, detailed_df)
        
        # Create advanced visualizations
        create_advanced_visualizations(summary, detailed_df, args.cv_output_dir)
        
        # Generate recommendations
        generate_recommendations(summary, detailed_df)
        
        print(f"\n ANALYSIS COMPLETE!")
        print(f"   Results directory: {args.cv_output_dir}")
        print(f"   New visualizations: comprehensive_cv_analysis.png")
        
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print(f"Make sure the cross-validation has completed successfully.")
        
    except Exception as e:
        print(f"UNEXPECTED ERROR: {e}")

if __name__ == "__main__":
    main() 