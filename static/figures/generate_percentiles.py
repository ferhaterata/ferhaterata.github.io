"""Generate percentiles and quantiles visualization."""
import numpy as np
import matplotlib.pyplot as plt

# Set style to match other figures
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['figure.facecolor'] = 'white'

# Generate sample data (normal distribution)
np.random.seed(42)
data = np.random.normal(100, 15, 1000)

# Calculate key percentiles
Q1, median, Q3 = np.percentile(data, [25, 50, 75])
P5, P95 = np.percentile(data, [5, 95])

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# ═══════════════════════════════════════════════════════════════
# Left: Histogram with key percentile lines
# ═══════════════════════════════════════════════════════════════
ax1 = axes[0]

# Histogram
n, bins, patches = ax1.hist(data, bins=35, color='#4A90D9', alpha=0.7, 
                            edgecolor='white', linewidth=0.5, density=True)

# Add percentile lines with labels
ax1.axvline(P5, color='#E74C3C', linestyle='--', linewidth=2, label=f'P5 = {P5:.1f}')
ax1.axvline(Q1, color='#F39C12', linestyle='-', linewidth=2, label=f'Q1 (P25) = {Q1:.1f}')
ax1.axvline(median, color='#27AE60', linestyle='-', linewidth=2.5, label=f'Median (P50) = {median:.1f}')
ax1.axvline(Q3, color='#F39C12', linestyle='-', linewidth=2, label=f'Q3 (P75) = {Q3:.1f}')
ax1.axvline(P95, color='#E74C3C', linestyle='--', linewidth=2, label=f'P95 = {P95:.1f}')

# Shade IQR region
ax1.axvspan(Q1, Q3, alpha=0.15, color='#F39C12', label=f'IQR = {Q3-Q1:.1f}')

ax1.set_xlabel('Value')
ax1.set_ylabel('Density')
ax1.set_title('Distribution with Key Percentiles')
ax1.legend(loc='upper right', fontsize=9, framealpha=0.95)
ax1.set_xlim(40, 160)

# ═══════════════════════════════════════════════════════════════
# Right: CDF with percentile reading demonstration
# ═══════════════════════════════════════════════════════════════
ax2 = axes[1]

# Compute empirical CDF
sorted_data = np.sort(data)
cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)

# Plot CDF
ax2.plot(sorted_data, cdf * 100, color='#4A90D9', linewidth=2.5)

# Mark quartiles on CDF
for percentile, value, color, name in [
    (25, Q1, '#F39C12', 'Q1'),
    (50, median, '#27AE60', 'Median'),
    (75, Q3, '#F39C12', 'Q3')
]:
    ax2.hlines(percentile, 40, value, colors=color, linestyles=':', linewidth=1.5, alpha=0.8)
    ax2.vlines(value, 0, percentile, colors=color, linestyles=':', linewidth=1.5, alpha=0.8)
    ax2.scatter([value], [percentile], color=color, s=60, zorder=5, edgecolor='white', linewidth=1)
    ax2.annotate(f'{name}\n({value:.0f})', xy=(value, percentile), 
                 xytext=(value+8, percentile-5), fontsize=9, color=color)

ax2.set_xlabel('Value')
ax2.set_ylabel('Cumulative Percentage (%)')
ax2.set_title('Cumulative Distribution Function (CDF)')
ax2.set_ylim(0, 100)
ax2.set_xlim(40, 160)
ax2.set_yticks([0, 25, 50, 75, 100])
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('percentiles_quantiles.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()

print("Generated: percentiles_quantiles.png")
