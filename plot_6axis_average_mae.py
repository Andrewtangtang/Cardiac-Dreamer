import json
import matplotlib.pyplot as plt
import numpy as np

# Read the JSON file
with open('results_cv_direct_backbone_eval/cv_direct_backbone_summary.json', 'r') as f:
    data = json.load(f)

# Extract MAE data for each axis across all patients
mae_data = {
    'x': [],
    'y': [],
    'z': [],
    'roll': [],
    'yaw': [],
    'pitch': []
}

for fold in data['fold_details']:
    mae_data['x'].append(fold['mae_x'])
    mae_data['y'].append(fold['mae_y'])
    mae_data['z'].append(fold['mae_z'])
    mae_data['roll'].append(fold['mae_roll'])
    mae_data['yaw'].append(fold['mae_yaw'])
    mae_data['pitch'].append(fold['mae_pitch'])

# Calculate average MAE for each axis
avg_mae = {}
std_mae = {}
for axis in mae_data:
    avg_mae[axis] = np.mean(mae_data[axis])
    std_mae[axis] = np.std(mae_data[axis])

# Prepare data for plotting
translation_axes = ['X', 'Y', 'Z']
rotation_axes = ['Roll', 'Yaw', 'Pitch']

translation_values = [avg_mae['x'], avg_mae['y'], avg_mae['z']]
translation_stds = [std_mae['x'], std_mae['y'], std_mae['z']]

rotation_values = [avg_mae['roll'], avg_mae['yaw'], avg_mae['pitch']]
rotation_stds = [std_mae['roll'], std_mae['yaw'], std_mae['pitch']]

# Create the plot with dual y-axis
fig, ax1 = plt.subplots(figsize=(12, 8))

# Plot translation axes (left y-axis)
x_pos_trans = np.arange(len(translation_axes))
bars1 = ax1.bar(x_pos_trans, translation_values, 0.6, 
                label='Translation (mm)', color='#2E8B8B', 
                alpha=0.8, edgecolor='black', linewidth=1)

# Add error bars for translation
ax1.errorbar(x_pos_trans, translation_values, yerr=translation_stds,
             fmt='none', color='black', capsize=5, linewidth=2)

# Add value labels on translation bars
for i, (bar, val) in enumerate(zip(bars1, translation_values)):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + translation_stds[i] + 0.2,
             f'{val:.2f}', ha='center', va='bottom', 
             fontweight='bold', fontsize=14, color='black')

# Set up left y-axis (translation)
ax1.set_xlabel('Axes', fontsize=18, fontweight='bold')
ax1.set_ylabel('Translation MAE (mm)', fontsize=18, fontweight='bold', color='#2E8B8B')
ax1.tick_params(axis='y', labelcolor='#2E8B8B', labelsize=16)
ax1.tick_params(axis='x', labelsize=16)
ax1.set_ylim(0, max([val + std for val, std in zip(translation_values, translation_stds)]) * 1.2)  # Adjust for error bars

# Create second y-axis for rotation
ax2 = ax1.twinx()

# Plot rotation axes (right y-axis)
x_pos_rot = np.arange(len(translation_axes), len(translation_axes) + len(rotation_axes))
bars2 = ax2.bar(x_pos_rot, rotation_values, 0.6,
                label='Rotation (degrees)', color='#D2691E',
                alpha=0.8, edgecolor='black', linewidth=1)

# Add error bars for rotation
ax2.errorbar(x_pos_rot, rotation_values, yerr=rotation_stds,
             fmt='none', color='black', capsize=5, linewidth=2)

# Add value labels on rotation bars
for i, (bar, val) in enumerate(zip(bars2, rotation_values)):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + rotation_stds[i] + 1.0,
             f'{val:.1f}', ha='center', va='bottom',
             fontweight='bold', fontsize=14, color='black')

# Set up right y-axis (rotation)
ax2.set_ylabel('Rotation MAE (degrees)', fontsize=18, fontweight='bold', color='#D2691E')
ax2.tick_params(axis='y', labelcolor='#D2691E', labelsize=16)
ax2.set_ylim(0, max([val + std for val, std in zip(rotation_values, rotation_stds)]) * 1.2)  # Adjust for error bars

# Set x-axis labels and positions - centered under each bar
all_axes = translation_axes + rotation_axes
all_positions = list(x_pos_trans) + list(x_pos_rot)
ax1.set_xticks(all_positions)
ax1.set_xticklabels(all_axes, fontsize=20, fontweight='bold')

# Add title
plt.title('Average 6-Axis MAE Across 5 Patients\n(Cross-Validation Results)', 
          fontsize=20, fontweight='bold', pad=25)

# Add legends
ax1.legend(loc='upper left', fontsize=14, frameon=True, fancybox=True, shadow=True)
ax2.legend(loc='upper right', fontsize=14, frameon=True, fancybox=True, shadow=True)

# Add grid
ax1.grid(True, alpha=0.3, axis='y', linestyle='--')

# Adjust layout
plt.tight_layout()

# Save the plot
plt.savefig('6axis_average_mae.png', dpi=300, bbox_inches='tight')
plt.savefig('6axis_average_mae.pdf', bbox_inches='tight')

# Show the plot
plt.show()

# Print detailed summary
print("6-Axis Average MAE Summary (5 Patients):")
print("=" * 50)
print("\nTranslation Axes (mm):")
for i, axis in enumerate(['X', 'Y', 'Z']):
    print(f"  {axis:5}: {translation_values[i]:6.3f}")

print("\nRotation Axes (degrees):")
for i, axis in enumerate(['Roll', 'Yaw', 'Pitch']):
    print(f"  {axis:5}: {rotation_values[i]:6.2f}")

print(f"\nOverall Translation Average: {np.mean(translation_values):.3f} mm")
print(f"Overall Rotation Average: {np.mean(rotation_values):.2f} degrees")

# Print individual patient data for reference
print("\n" + "="*50)
print("Individual Patient Data:")
for i in range(5):
    print(f"\nPatient {i+1}:")
    print(f"  Translation: X={mae_data['x'][i]:.2f}, Y={mae_data['y'][i]:.2f}, Z={mae_data['z'][i]:.2f} mm")
    print(f"  Rotation: Roll={mae_data['roll'][i]:.1f}, Yaw={mae_data['yaw'][i]:.1f}, Pitch={mae_data['pitch'][i]:.1f} deg") 