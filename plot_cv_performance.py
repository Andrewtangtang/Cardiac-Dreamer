import json
import matplotlib.pyplot as plt
import numpy as np

# Read the JSON file
with open('results_cv_direct_backbone_eval/cv_direct_backbone_summary.json', 'r') as f:
    data = json.load(f)

# Extract correlation data for each fold/patient
total_losses = []
fold_labels = []

for fold in data['fold_details']:
    total_losses.append(fold['total_loss'])
    fold_labels.append(f"Patient {fold['fold_num']}")

# Calculate mean and std
mean_loss = np.mean(total_losses)
std_loss = np.std(total_losses)

# Create the plot
fig, ax = plt.subplots(figsize=(8, 6))

# Create bars
bars = ax.bar(range(1, len(total_losses) + 1), total_losses, 
              color='lightblue', edgecolor='black', linewidth=1, alpha=0.8)

# Add value labels on top of bars
for i, (bar, loss) in enumerate(zip(bars, total_losses)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{loss:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=12)

# Add horizontal line for mean
ax.axhline(y=mean_loss, color='red', linestyle='--', linewidth=2)

# Add text for mean and std
ax.text(len(total_losses) + 0.3, mean_loss + 0.02, 
        f'Mean: {mean_loss:.4f}', 
        ha='left', va='bottom', color='red', fontweight='bold', fontsize=12)

# Customize the plot
ax.set_xlabel('Patient', fontsize=16, fontweight='bold')
ax.set_ylabel('Total Loss', fontsize=16, fontweight='bold')
ax.set_title('Cross-Validation Performance by Patient', fontsize=18, fontweight='bold', pad=20)

# Set x-axis
ax.set_xticks(range(1, len(total_losses) + 1))
ax.set_xticklabels([f'{i}' for i in range(1, len(total_losses) + 1)], fontsize=14)

# Set y-axis tick labels
ax.tick_params(axis='y', labelsize=14)

# Set y-axis limits
ax.set_ylim(0, max(total_losses) * 1.2)

# Add grid
ax.grid(True, alpha=0.3, axis='y')

# Adjust layout
plt.tight_layout()

# Save the plot
plt.savefig('cv_performance_by_patient.png', dpi=300, bbox_inches='tight')
plt.savefig('cv_performance_by_patient.pdf', bbox_inches='tight')

# Show the plot
plt.show()

# Print summary
print("Cross-Validation Performance Summary:")
print("=" * 40)
for i, loss in enumerate(total_losses, 1):
    print(f"Patient {i}: {loss:.4f}")
print(f"\nMean ± STD: {mean_loss:.4f} ± {std_loss:.4f}") 