"""
Generate visualization of Internal Covariate Shift.
Shows how activation distributions change during training, causing optimization difficulties.
"""

import numpy as np
import matplotlib.pyplot as plt

def norm_pdf(x, mean, std):
    """Normal distribution PDF using numpy (no scipy dependency)."""
    return (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Common x range
x = np.linspace(-0.5, 1.5, 500)

# Distribution parameters
# Step 1: centered distribution
mean1, std1 = 0.5, 0.3
# Step 100: shifted and narrower distribution  
mean2, std2 = 0.8, 0.1


# Colors
color1 = '#2196F3'  # Blue
color2 = '#F44336'  # Red
fill_alpha = 0.3

# === Left Panel: Training Step 1 ===
ax1 = axes[0]
y1 = norm_pdf(x, mean1, std1)
ax1.plot(x, y1, color=color1, linewidth=2.5, label=f'μ={mean1}, σ={std1}')
ax1.fill_between(x, y1, alpha=fill_alpha, color=color1)
ax1.axvline(mean1, color=color1, linestyle='--', alpha=0.7, linewidth=1.5)

ax1.set_title('Training Step 1', fontsize=14, fontweight='bold')
ax1.set_xlabel('Activation Value')
ax1.set_ylabel('Density')
ax1.set_xlim(-0.3, 1.3)
ax1.set_ylim(0, 4.5)

# Annotation box
textstr1 = f'Input to Layer 3:\nmean = {mean1}\nstd = {std1}'
props = dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor=color1, alpha=0.9)
ax1.text(0.02, 0.98, textstr1, transform=ax1.transAxes, fontsize=10,
         verticalalignment='top', bbox=props)

# Add note about learned weights
ax1.text(0.5, -0.22, 'Layer 3 learns weights\noptimized for THIS distribution', 
         transform=ax1.transAxes, fontsize=10, ha='center',
         style='italic', color='#333333')

ax1.legend(loc='upper right', fontsize=10)

# === Right Panel: Training Step 100 ===
ax2 = axes[1]
y2 = norm_pdf(x, mean2, std2)
ax2.plot(x, y2, color=color2, linewidth=2.5, label=f'μ={mean2}, σ={std2}')
ax2.fill_between(x, y2, alpha=fill_alpha, color=color2)
ax2.axvline(mean2, color=color2, linestyle='--', alpha=0.7, linewidth=1.5)

# Show ghost of original distribution for comparison
ax2.plot(x, y1, color=color1, linewidth=1.5, linestyle=':', alpha=0.5, label='Original (step 1)')
ax2.fill_between(x, y1, alpha=0.1, color=color1)

ax2.set_title('Training Step 100', fontsize=14, fontweight='bold')
ax2.set_xlabel('Activation Value')
ax2.set_ylabel('Density')
ax2.set_xlim(-0.3, 1.3)
ax2.set_ylim(0, 4.5)

# Annotation box
textstr2 = f'Input to Layer 3:\nmean = {mean2}\nstd = {std2}'
props2 = dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor=color2, alpha=0.9)
ax2.text(0.02, 0.98, textstr2, transform=ax2.transAxes, fontsize=10,
         verticalalignment='top', bbox=props2)

# Add note about wrong weights
ax2.text(0.5, -0.22, "Layer 3's weights are now\nWRONG for the shifted input!", 
         transform=ax2.transAxes, fontsize=10, ha='center',
         style='italic', color='#c62828', fontweight='bold')

ax2.legend(loc='upper right', fontsize=10)

# Add text between panels showing the shift
fig.text(0.5, 0.5, '→', fontsize=40, ha='center', va='center', 
         color='#333333', fontweight='bold')
fig.text(0.5, 0.42, 'Distribution\nShift', fontsize=10, ha='center', va='top',
         color='#666666', style='italic')

# Main title
fig.suptitle('Internal Covariate Shift: Why BatchNorm Helps', 
             fontsize=16, fontweight='bold', y=1.02)

plt.tight_layout()
plt.subplots_adjust(bottom=0.18)  # Make room for bottom text annotations
plt.savefig('internal_covariate_shift.png', dpi=150, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.close()

print("Generated: internal_covariate_shift.png")
