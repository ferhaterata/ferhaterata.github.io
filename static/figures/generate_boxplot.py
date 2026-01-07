"""
Generate box-plot anatomy figure showing outlier detection concepts.
"""
import matplotlib.pyplot as plt
import numpy as np

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13

# Generate sample data with outliers
np.random.seed(42)
data = np.concatenate([
    np.random.normal(50, 10, 100),  # Main distribution
    np.array([10, 15, 95, 100, 105])  # Outliers
])

# Calculate statistics
Q1 = np.percentile(data, 25)
Q2 = np.percentile(data, 50)  # Median
Q3 = np.percentile(data, 75)
IQR = Q3 - Q1
lower_whisker = Q1 - 1.5 * IQR
upper_whisker = Q3 + 1.5 * IQR

# Find outliers
outliers = data[(data < lower_whisker) | (data > upper_whisker)]
non_outliers = data[(data >= lower_whisker) & (data <= upper_whisker)]

# Create figure with 2 subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# ======================
# Left: Annotated Box Plot
# ======================
ax1 = axes[0]
bp = ax1.boxplot([data], vert=True, widths=0.5, patch_artist=True,
                  flierprops=dict(marker='o', markerfacecolor='red', markersize=8, alpha=0.7))

# Color the box
bp['boxes'][0].set_facecolor('#3498db')
bp['boxes'][0].set_alpha(0.5)
bp['medians'][0].set_color('darkblue')
bp['medians'][0].set_linewidth(2)

# Add annotations
ax1.annotate(f'Median (Q2) = {Q2:.1f}', xy=(1, Q2), xytext=(1.4, Q2),
            fontsize=10, ha='left', va='center',
            arrowprops=dict(arrowstyle='->', color='darkblue'))

ax1.annotate(f'Q3 = {Q3:.1f}', xy=(1.25, Q3), xytext=(1.4, Q3 + 3),
            fontsize=10, ha='left', va='center',
            arrowprops=dict(arrowstyle='->', color='gray'))

ax1.annotate(f'Q1 = {Q1:.1f}', xy=(1.25, Q1), xytext=(1.4, Q1 - 3),
            fontsize=10, ha='left', va='center',
            arrowprops=dict(arrowstyle='->', color='gray'))

# Annotate whiskers
actual_upper_whisker = min(upper_whisker, non_outliers.max())
actual_lower_whisker = max(lower_whisker, non_outliers.min())

ax1.annotate(f'Upper Whisker\n(Q3 + 1.5×IQR)', 
            xy=(1, actual_upper_whisker), xytext=(1.4, actual_upper_whisker + 8),
            fontsize=9, ha='left', va='bottom',
            arrowprops=dict(arrowstyle='->', color='green'))

ax1.annotate(f'Lower Whisker\n(Q1 - 1.5×IQR)', 
            xy=(1, actual_lower_whisker), xytext=(1.4, actual_lower_whisker - 8),
            fontsize=9, ha='left', va='top',
            arrowprops=dict(arrowstyle='->', color='green'))

# Annotate outliers
if len(outliers) > 0:
    ax1.annotate('Outliers\n(beyond whiskers)', 
                xy=(1, outliers.max()), xytext=(0.55, outliers.max() + 5),
                fontsize=9, ha='center', va='bottom', color='red',
                arrowprops=dict(arrowstyle='->', color='red'))

# Add IQR bracket
ax1.annotate('', xy=(0.7, Q1), xytext=(0.7, Q3),
            arrowprops=dict(arrowstyle='<->', color='purple', lw=2))
ax1.text(0.62, (Q1 + Q3)/2, f'IQR\n{IQR:.1f}', fontsize=9, ha='right', va='center', color='purple')

ax1.set_xlim(0.4, 2.0)
ax1.set_ylabel('Value')
ax1.set_xticklabels(['Data'])
ax1.set_title('Anatomy of a Box Plot', fontweight='bold')

# ======================
# Right: IQR Method for Outlier Detection
# ======================
ax2 = axes[1]

# Show the distribution with histogram
ax2_hist = ax2.twinx()
ax2_hist.hist(data, bins=20, alpha=0.3, color='blue', edgecolor='black')
ax2_hist.set_ylabel('Count', color='blue')
ax2_hist.tick_params(axis='y', labelcolor='blue')

# Mark regions
x_range = np.linspace(0, 120, 1000)
ax2.axvline(lower_whisker, color='red', linestyle='--', linewidth=2, label=f'Lower bound: {lower_whisker:.1f}')
ax2.axvline(upper_whisker, color='red', linestyle='--', linewidth=2, label=f'Upper bound: {upper_whisker:.1f}')
ax2.axvline(Q1, color='orange', linestyle='-', linewidth=1.5, alpha=0.7, label=f'Q1: {Q1:.1f}')
ax2.axvline(Q3, color='orange', linestyle='-', linewidth=1.5, alpha=0.7, label=f'Q3: {Q3:.1f}')
ax2.axvline(Q2, color='blue', linestyle='-', linewidth=2, label=f'Median: {Q2:.1f}')

# Shade outlier regions
ax2.axvspan(0, lower_whisker, alpha=0.2, color='red', label='Outlier region')
ax2.axvspan(upper_whisker, 120, alpha=0.2, color='red')

# Mark outliers
ax2.scatter(outliers, [0.5]*len(outliers), color='red', s=100, zorder=5, marker='X', label='Outliers detected')

ax2.set_xlim(0, 120)
ax2.set_ylim(0, 1)
ax2.set_xlabel('Value')
ax2.set_yticks([])
ax2.legend(loc='upper right', fontsize=8)
ax2.set_title('IQR Method: Outlier Detection', fontweight='bold')

# Add formula box
formula_text = (
    "IQR Rule for Outliers:\n"
    "─────────────────────\n"
    f"IQR = Q3 - Q1 = {Q3:.1f} - {Q1:.1f} = {IQR:.1f}\n\n"
    f"Lower bound = Q1 - 1.5×IQR = {lower_whisker:.1f}\n"
    f"Upper bound = Q3 + 1.5×IQR = {upper_whisker:.1f}\n\n"
    "Points outside bounds → Outliers"
)
ax2.text(0.02, 0.98, formula_text, transform=ax2.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.savefig('boxplot_anatomy.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

print("Generated: boxplot_anatomy.png")
print(f"\nStatistics:")
print(f"  Q1 (25th percentile): {Q1:.2f}")
print(f"  Q2 (Median): {Q2:.2f}")
print(f"  Q3 (75th percentile): {Q3:.2f}")
print(f"  IQR: {IQR:.2f}")
print(f"  Lower whisker: {lower_whisker:.2f}")
print(f"  Upper whisker: {upper_whisker:.2f}")
print(f"  Outliers detected: {len(outliers)} points")
