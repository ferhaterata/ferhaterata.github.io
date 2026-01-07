"""
Generate Confidence Intervals and Percentiles Visualization
Shows repeated CI simulation, percentiles on distribution, and CI width factors
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Set random seed for reproducibility
np.random.seed(42)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# ============================================
# Left panel: Repeated CI Simulation
# ============================================
ax = axes[0]

true_mean = 100
true_std = 15
n_samples = 30
n_experiments = 25
confidence_level = 0.95
z_star = stats.norm.ppf(1 - (1 - confidence_level) / 2)

# Simulate experiments
misses = 0
for i in range(n_experiments):
    # Draw a sample
    sample = np.random.normal(true_mean, true_std, n_samples)
    sample_mean = np.mean(sample)
    sample_std = np.std(sample, ddof=1)
    
    # Calculate CI
    margin = z_star * sample_std / np.sqrt(n_samples)
    ci_lower = sample_mean - margin
    ci_upper = sample_mean + margin
    
    # Check if CI contains true mean
    contains_true = ci_lower <= true_mean <= ci_upper
    color = 'blue' if contains_true else 'red'
    linewidth = 1.5 if contains_true else 2.5
    
    if not contains_true:
        misses += 1
    
    # Plot CI
    ax.plot([ci_lower, ci_upper], [i, i], color=color, linewidth=linewidth, solid_capstyle='round')
    ax.plot(sample_mean, i, 'o', color=color, markersize=4)

# True mean line
ax.axvline(true_mean, color='green', linewidth=2.5, linestyle='--', label=f'True mean = {true_mean}')

ax.set_xlabel('Value')
ax.set_ylabel('Experiment Number')
ax.set_title(f'95% Confidence Intervals from {n_experiments} Experiments\n({misses} intervals miss the true mean)', fontweight='bold')
ax.legend(loc='upper right', fontsize=9)

# Add annotation
textstr = f'"95% confident" means:\n~95% of intervals contain\nthe true mean\n(Here: {n_experiments - misses}/{n_experiments} = {100*(n_experiments-misses)/n_experiments:.0f}%)'
props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.8)
ax.text(0.02, 0.02, textstr, transform=ax.transAxes, fontsize=9,
        verticalalignment='bottom', bbox=props)

# Color legend
ax.plot([], [], 'b-', linewidth=2, label='Contains true mean')
ax.plot([], [], 'r-', linewidth=2, label='Misses true mean')
ax.legend(loc='upper right', fontsize=9)

# ============================================
# Middle panel: Percentiles on Distribution
# ============================================
ax = axes[1]

x = np.linspace(-4, 4, 500)
y = stats.norm.pdf(x)

# Plot distribution
ax.plot(x, y, 'b-', linewidth=2.5)
ax.fill_between(x, y, alpha=0.15, color='blue')

# Key percentiles for standard normal
percentiles = {
    'P1': stats.norm.ppf(0.01),
    'P5': stats.norm.ppf(0.05),
    'Q1 (P25)': stats.norm.ppf(0.25),
    'Median': stats.norm.ppf(0.50),
    'Q3 (P75)': stats.norm.ppf(0.75),
    'P95': stats.norm.ppf(0.95),
    'P99': stats.norm.ppf(0.99),
}

colors = ['darkred', 'red', 'orange', 'green', 'orange', 'red', 'darkred']
for (name, val), color in zip(percentiles.items(), colors):
    ax.axvline(val, color=color, linewidth=1.5, linestyle='--', alpha=0.8)
    # Position labels
    y_pos = stats.norm.pdf(val) + 0.02
    if 'P1' in name or 'P99' in name:
        y_pos = 0.35
    elif 'P5' in name or 'P95' in name:
        y_pos = 0.32
    elif 'Q1' in name or 'Q3' in name:
        y_pos = 0.29
    else:
        y_pos = 0.38
    
    ax.annotate(name, xy=(val, y_pos), fontsize=8, ha='center', color=color, fontweight='bold')

# Shade IQR
q1 = stats.norm.ppf(0.25)
q3 = stats.norm.ppf(0.75)
x_iqr = x[(x >= q1) & (x <= q3)]
ax.fill_between(x_iqr, stats.norm.pdf(x_iqr), color='green', alpha=0.3, label='IQR (middle 50%)')

ax.set_xlabel('Standard Deviations from Mean')
ax.set_ylabel('Probability Density')
ax.set_title('Percentiles on Standard Normal Distribution', fontweight='bold')
ax.legend(loc='upper left', fontsize=9)
ax.set_xlim(-4, 4)
ax.set_ylim(0, 0.45)

# Add percentile interpretation
textstr = 'Percentile = value below\nwhich p% of data falls\n\nP95 = 95th percentile:\n95% of data below this value'
props = dict(boxstyle='round', facecolor='lightblue', alpha=0.5)
ax.text(0.98, 0.98, textstr, transform=ax.transAxes, fontsize=8,
        verticalalignment='top', ha='right', bbox=props)

# ============================================
# Right panel: CI Width Factors
# ============================================
ax = axes[2]

# Show how CI width changes with n and confidence level
sample_sizes = np.array([10, 25, 50, 100, 200, 400])
sigma = 15

# Different confidence levels
confidence_levels = [0.90, 0.95, 0.99]
colors_conf = ['green', 'blue', 'red']
linestyles = ['-', '-', '-']

for cl, color, ls in zip(confidence_levels, colors_conf, linestyles):
    z = stats.norm.ppf(1 - (1 - cl) / 2)
    ci_widths = 2 * z * sigma / np.sqrt(sample_sizes)
    ax.plot(sample_sizes, ci_widths, color=color, linewidth=2.5, linestyle=ls, 
            marker='o', markersize=6, label=f'{int(cl*100)}% CI')

ax.set_xlabel('Sample Size (n)')
ax.set_ylabel('CI Width (2 * margin of error)')
ax.set_title('CI Width Decreases with More Data\n(but diminishing returns!)', fontweight='bold')
ax.legend(loc='upper right', fontsize=10)
ax.set_xlim(0, 420)

# Add annotation about sqrt(n)
ax.annotate('Width ~ 1/sqrt(n)\n\nDouble precision?\nNeed 4x the data!', 
            xy=(250, 15), fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Add grid lines
ax.grid(True, alpha=0.3)

# Add secondary x-axis showing relative width
ax2 = ax.twiny()
ax2.set_xlim(ax.get_xlim())
ax2.set_xticks([10, 40, 100, 400])
ax2.set_xticklabels(['1x', '2x', '3.2x', '6.3x'])
ax2.set_xlabel('Relative precision improvement', fontsize=9)

plt.tight_layout()
plt.savefig('ml-notes/figures/confidence_intervals_percentiles.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()

print("Generated: confidence_intervals_percentiles.png")
