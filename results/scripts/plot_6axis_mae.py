import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Read the JSON file
with open('results_cv_direct_backbone_eval/cv_direct_backbone_summary.json', 'r') as f:
    data = json.load(f)

# Extract data for each patient (fold)
patients = []
mae_data = []

for fold in data['fold_details']:
    patient_name = fold['val_patient_group']
    patients.append(patient_name)
    
    # Extract overall translation and rotation MAE values
    mae_values = [
        fold['translation_mae'],  # Overall XYZ
        fold['mae_roll'],
        fold['mae_yaw'],
        fold['mae_pitch']
    ]
    mae_data.append(mae_values)

# Convert to numpy array for easier handling
mae_data = np.array(mae_data)

# Axis labels
axis_labels = ['Translation\n(XYZ)', 'Roll', 'Yaw', 'Pitch']

# Create the plot
fig, ax = plt.subplots(figsize=(10, 8))

# Set the width of bars and positions
bar_width = 0.15
positions = np.arange(len(axis_labels))

# Colors for each patient
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']

# Plot bars for each patient
for i, patient in enumerate(patients):
    offset = (i - 2) * bar_width  # Center the bars
    bars = ax.bar(positions + offset, mae_data[i], bar_width, 
                  label=patient, color=colors[i], alpha=0.8)
    
    # Add value labels on top of bars
    for j, bar in enumerate(bars):
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),  # 3 points vertical offset
                   textcoords="offset points",
                   ha='center', va='bottom',
                   fontsize=8, rotation=0)

# Customize the plot
ax.set_xlabel('Motion Components', fontsize=12, fontweight='bold')
ax.set_ylabel('Mean Absolute Error (MAE)', fontsize=12, fontweight='bold')
ax.set_title('6-Axis MAE Comparison Across Patients\n(Evaluated via Cross-Validation on Each Subject)', 
             fontsize=14, fontweight='bold', pad=20)

# Set x-axis labels
ax.set_xticks(positions)
ax.set_xticklabels(axis_labels)

# Add legend
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

# Add grid for better readability
ax.grid(True, alpha=0.3, axis='y')

# Adjust layout to prevent legend cutoff
plt.tight_layout()

# Save the plot
plt.savefig('overall_mae_comparison.png', dpi=300, bbox_inches='tight')
plt.savefig('overall_mae_comparison.pdf', bbox_inches='tight')

# Show the plot
plt.show()

# Also create a summary table
print("\nOverall MAE Summary Table:")
print("=" * 60)
df = pd.DataFrame(mae_data, 
                  index=patients, 
                  columns=['Translation(XYZ)', 'Roll', 'Yaw', 'Pitch'])
print(df.round(3))

# Calculate and print statistics
print(f"\nStatistics across all patients:")
print("=" * 40)
print(f"Mean MAE per component:")
component_names = ['Translation(XYZ)', 'Roll', 'Yaw', 'Pitch']
for i, component in enumerate(component_names):
    mean_mae = np.mean(mae_data[:, i])
    std_mae = np.std(mae_data[:, i])
    print(f"  {component:15}: {mean_mae:6.3f} ± {std_mae:.3f}")

print(f"\nTranslation average: {np.mean(mae_data[:, 0]):.3f}")
print(f"Rotation average: {np.mean(mae_data[:, 1:]):.3f}") 