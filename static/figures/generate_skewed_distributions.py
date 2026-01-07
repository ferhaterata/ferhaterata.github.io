"""Generate figure showing Mean, Median, Mode for symmetric, right-skewed, and left-skewed distributions."""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Set style
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['figure.facecolor'] = 'white'

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# Generate data for each distribution type
np.random.seed(42)

# 1. Symmetric (Normal)
x_sym = np.linspace(-4, 4, 1000)
y_sym = stats.norm.pdf(x_sym, 0, 1)
mean_sym = 0
median_sym = 0
mode_sym = 0

# 2. Right-skewed (Log-normal / Chi-squared)
x_right = np.linspace(0, 10, 1000)
y_right = stats.chi2.pdf(x_right, df=3)
samples_right = stats.chi2.rvs(df=3, size=100000)
mean_right = np.mean(samples_right)
median_right = np.median(samples_right)
mode_right = x_right[np.argmax(y_right)]

# 3. Left-skewed (Reflected exponential / Beta)
x_left = np.linspace(0, 1, 1000)
y_left = stats.beta.pdf(x_left, a=5, b=2)
samples_left = stats.beta.rvs(a=5, b=2, size=100000)
mean_left = np.mean(samples_left)
median_left = np.median(samples_left)
mode_left = x_left[np.argmax(y_left)]

# Colors
color_mode = '#2ecc71'    # Green
color_median = '#e74c3c'  # Red
color_mean = '#3498db'    # Blue
line_alpha = 0.9

# Plot 1: Symmetric
ax1 = axes[0]
ax1.fill_between(x_sym, y_sym, alpha=0.3, color='gray')
ax1.plot(x_sym, y_sym, 'k-', linewidth=2)
ax1.axvline(mode_sym, color=color_mode, linestyle='-', linewidth=2.5, label=f'Mode = {mode_sym:.1f}')
ax1.axvline(median_sym, color=color_median, linestyle='--', linewidth=2.5, label=f'Median = {median_sym:.1f}')
ax1.axvline(mean_sym, color=color_mean, linestyle=':', linewidth=2.5, label=f'Mean = {mean_sym:.1f}')
ax1.set_title('Symmetric Distribution\n(Mean = Median = Mode)', fontweight='bold')
ax1.set_xlabel('Value')
ax1.set_ylabel('Density')
ax1.legend(loc='upper right', fontsize=9)
ax1.set_xlim(-4, 4)
ax1.set_ylim(0, 0.45)
ax1.grid(True, alpha=0.3)

# Plot 2: Right-skewed
ax2 = axes[1]
ax2.fill_between(x_right, y_right, alpha=0.3, color='gray')
ax2.plot(x_right, y_right, 'k-', linewidth=2)
ax2.axvline(mode_right, color=color_mode, linestyle='-', linewidth=2.5, label=f'Mode = {mode_right:.1f}')
ax2.axvline(median_right, color=color_median, linestyle='--', linewidth=2.5, label=f'Median = {median_right:.1f}')
ax2.axvline(mean_right, color=color_mean, linestyle=':', linewidth=2.5, label=f'Mean = {mean_right:.1f}')
ax2.set_title('Right-Skewed Distribution\n(Mode < Median < Mean)', fontweight='bold')
ax2.set_xlabel('Value')
ax2.set_ylabel('Density')
ax2.legend(loc='upper right', fontsize=9)
ax2.set_xlim(0, 10)
ax2.grid(True, alpha=0.3)

# Add tail arrow annotation
ax2.annotate('', xy=(9, 0.02), xytext=(6, 0.02),
            arrowprops=dict(arrowstyle='->', color='#7f8c8d', lw=2))
ax2.text(7.5, 0.035, 'tail pulls\nmean right', ha='center', fontsize=9, color='#7f8c8d')

# Plot 3: Left-skewed
ax3 = axes[2]
ax3.fill_between(x_left, y_left, alpha=0.3, color='gray')
ax3.plot(x_left, y_left, 'k-', linewidth=2)
ax3.axvline(mode_left, color=color_mode, linestyle='-', linewidth=2.5, label=f'Mode = {mode_left:.2f}')
ax3.axvline(median_left, color=color_median, linestyle='--', linewidth=2.5, label=f'Median = {median_left:.2f}')
ax3.axvline(mean_left, color=color_mean, linestyle=':', linewidth=2.5, label=f'Mean = {mean_left:.2f}')
ax3.set_title('Left-Skewed Distribution\n(Mean < Median < Mode)', fontweight='bold')
ax3.set_xlabel('Value')
ax3.set_ylabel('Density')
ax3.legend(loc='upper left', fontsize=9)
ax3.set_xlim(0, 1)
ax3.grid(True, alpha=0.3)

# Add tail arrow annotation
ax3.annotate('', xy=(0.1, 0.1), xytext=(0.4, 0.1),
            arrowprops=dict(arrowstyle='->', color='#7f8c8d', lw=2))
ax3.text(0.25, 0.18, 'tail pulls\nmean left', ha='center', fontsize=9, color='#7f8c8d')

plt.tight_layout()
plt.savefig('skewed_distributions.png', dpi=150, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.close()

print("Generated: skewed_distributions.png")
