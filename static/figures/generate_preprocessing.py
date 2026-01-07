#!/usr/bin/env python3
"""Generate data preprocessing before/after visualization."""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.size'] = 11

np.random.seed(42)

# Create data with very different scales (common real-world scenario)
n_samples = 100

# Feature 1: Age (0-80)
age = np.random.normal(40, 15, n_samples).clip(18, 80)

# Feature 2: Salary (20,000 - 500,000)
salary = np.random.exponential(80000, n_samples).clip(20000, 500000)

# Feature 3: Height in meters (1.5 - 2.0)
height = np.random.normal(1.7, 0.1, n_samples).clip(1.4, 2.1)

# Create figure
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Left: Before normalization (raw data)
ax1 = axes[0]
# Show all three features on same scale - salary dominates
x_pos = [1, 2, 3]
bp1 = ax1.boxplot([age, salary, height], positions=x_pos, widths=0.6, patch_artist=True)
colors = ['#ff9999', '#99ff99', '#9999ff']
for patch, color in zip(bp1['boxes'], colors):
    patch.set_facecolor(color)

ax1.set_xticks(x_pos)
ax1.set_xticklabels(['Age\n(18-80)', 'Salary\n($20K-500K)', 'Height\n(1.4-2.1m)'])
ax1.set_ylabel('Raw Value', fontsize=12)
ax1.set_title('Before Normalization\n(Different Scales!)', fontsize=13, fontweight='bold', color='red')
ax1.set_yscale('log')
ax1.grid(True, alpha=0.3, axis='y')

# Add annotation showing the problem
ax1.annotate('Salary dominates!\n~10,000x larger than height', 
             xy=(2, 100000), xytext=(2.5, 5),
             fontsize=10, ha='center',
             arrowprops=dict(arrowstyle='->', color='red'),
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# Middle: Standardization (z-score)
ax2 = axes[1]
age_std = (age - age.mean()) / age.std()
salary_std = (salary - salary.mean()) / salary.std()
height_std = (height - height.mean()) / height.std()

bp2 = ax2.boxplot([age_std, salary_std, height_std], positions=x_pos, widths=0.6, patch_artist=True)
for patch, color in zip(bp2['boxes'], colors):
    patch.set_facecolor(color)

ax2.set_xticks(x_pos)
ax2.set_xticklabels(['Age', 'Salary', 'Height'])
ax2.set_ylabel('Standardized Value (z-score)', fontsize=12)
ax2.set_title('After Standardization\n(Mean=0, Std=1)', fontsize=13, fontweight='bold', color='green')
ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
ax2.set_ylim(-4, 4)
ax2.grid(True, alpha=0.3)

# Add formula
ax2.text(2, 3.5, r'$x_{std} = \frac{x - \mu}{\sigma}$', fontsize=14, ha='center',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Right: Min-Max normalization
ax3 = axes[2]
age_mm = (age - age.min()) / (age.max() - age.min())
salary_mm = (salary - salary.min()) / (salary.max() - salary.min())
height_mm = (height - height.min()) / (height.max() - height.min())

bp3 = ax3.boxplot([age_mm, salary_mm, height_mm], positions=x_pos, widths=0.6, patch_artist=True)
for patch, color in zip(bp3['boxes'], colors):
    patch.set_facecolor(color)

ax3.set_xticks(x_pos)
ax3.set_xticklabels(['Age', 'Salary', 'Height'])
ax3.set_ylabel('Normalized Value [0, 1]', fontsize=12)
ax3.set_title('After Min-Max Normalization\n(Range [0, 1])', fontsize=13, fontweight='bold', color='green')
ax3.set_ylim(-0.1, 1.1)
ax3.axhline(y=0, color='black', linestyle='--', alpha=0.3)
ax3.axhline(y=1, color='black', linestyle='--', alpha=0.3)
ax3.grid(True, alpha=0.3)

# Add formula
ax3.text(2, 1.05, r'$x_{norm} = \frac{x - x_{min}}{x_{max} - x_{min}}$', fontsize=14, ha='center',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Add main explanation
fig.text(0.5, -0.03, 
         'Why normalize? (1) Gradient descent converges faster when features have similar scales\n'
         '(2) Features with large values don\'t dominate the loss  (3) Learning rate works for all features',
         ha='center', fontsize=11, style='italic', 
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.subplots_adjust(bottom=0.16)
plt.savefig('preprocessing_before_after.png', bbox_inches='tight', facecolor='white')
plt.close()

print("Generated preprocessing_before_after.png")
