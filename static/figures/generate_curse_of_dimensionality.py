#!/usr/bin/env python3
"""
Generate Curse of Dimensionality figures:
1. Hughes phenomenon - accuracy vs dimensionality
2. Distance concentration in high dimensions
"""

import numpy as np
import matplotlib.pyplot as plt

# Set style
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['font.size'] = 11

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ============================================================
# Panel 1: Hughes Phenomenon - Accuracy vs Dimensionality
# ============================================================
ax1 = axes[0]

# Simulated accuracy curves for different sample sizes
dimensions = np.arange(1, 101)

def hughes_curve(d, n, d_optimal, max_acc=0.95, min_acc=0.5):
    """Simulate Hughes phenomenon curve"""
    # Accuracy increases initially, peaks, then decreases
    # Peak earlier with fewer samples
    acc = max_acc - 0.3 * np.exp(-0.1 * d)  # Initial increase
    # Curse kicks in - overfitting after peak
    curse_factor = np.maximum(0, (d - d_optimal) / d_optimal) ** 1.5
    acc = acc - curse_factor * (max_acc - min_acc) * 0.5
    # Add noise for realism
    acc = np.clip(acc + np.random.randn(len(d)) * 0.01, min_acc, max_acc)
    return acc

np.random.seed(42)
# Different sample sizes
for n, color, d_opt in [(50, '#e74c3c', 8), (200, '#f39c12', 25), (1000, '#3498db', 50), (10000, '#2ecc71', 80)]:
    acc = hughes_curve(dimensions, n, d_opt)
    ax1.plot(dimensions, acc, color=color, linewidth=2.5, label=f'n = {n:,}')
    # Mark the peak
    peak_idx = np.argmax(acc[:60])  # Look for peak in first 60 dims
    ax1.scatter([dimensions[peak_idx]], [acc[peak_idx]], color=color, s=100, zorder=5, marker='o')

ax1.axvline(x=50, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
ax1.annotate('More samples\n→ curse delayed', xy=(75, 0.88), fontsize=10, ha='center',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

ax1.set_xlabel('Number of Dimensions (D)', fontsize=12)
ax1.set_ylabel('Classification Accuracy', fontsize=12)
ax1.set_title('Hughes Phenomenon: Accuracy Peaks Then Declines\n(The Curse of Dimensionality)', fontsize=13, fontweight='bold')
ax1.legend(title='Sample Size', loc='lower left')
ax1.set_xlim(0, 100)
ax1.set_ylim(0.5, 1.0)

# Add annotations
ax1.annotate('Useful features\nimprove accuracy', xy=(10, 0.7), xytext=(15, 0.58),
             fontsize=9, arrowprops=dict(arrowstyle='->', color='gray'),
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
ax1.annotate('Curse kicks in:\noverfitting', xy=(70, 0.65), xytext=(80, 0.58),
             fontsize=9, arrowprops=dict(arrowstyle='->', color='gray'),
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# ============================================================
# Panel 2: Distance Concentration
# ============================================================
ax2 = axes[1]

# Simulate distance ratio: max_dist / min_dist as dimension increases
dimensions_dist = np.array([1, 2, 5, 10, 20, 50, 100, 200, 500, 1000])
n_points = 100

distance_ratios = []
for d in dimensions_dist:
    # Generate random points in d-dimensional unit hypercube
    points = np.random.rand(n_points, d)
    # Compute all pairwise distances
    distances = []
    for i in range(n_points):
        for j in range(i+1, n_points):
            dist = np.sqrt(np.sum((points[i] - points[j])**2))
            distances.append(dist)
    distances = np.array(distances)
    # Ratio of max to min distance
    ratio = (distances.max() - distances.min()) / distances.min()
    distance_ratios.append(ratio)

ax2.plot(dimensions_dist, distance_ratios, 'o-', color='#9b59b6', linewidth=2.5, markersize=10)
ax2.set_xscale('log')
ax2.set_xlabel('Number of Dimensions (log scale)', fontsize=12)
ax2.set_ylabel('(Max - Min Distance) / Min Distance', fontsize=12)
ax2.set_title('Distance Concentration:\nAll Points Become Equidistant', fontsize=13, fontweight='bold')

# Add asymptotic line
ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=2, label='Ratio → 0 (equidistant)')
ax2.legend()

# Add annotation
ax2.annotate('In high D:\n"nearest" = "farthest"\n→ k-NN fails', xy=(500, 0.3),
             fontsize=10, ha='center',
             bbox=dict(boxstyle='round', facecolor='#f8e8f8', alpha=0.9))

plt.tight_layout()
plt.savefig('curse_of_dimensionality.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()

print("Generated: curse_of_dimensionality.png")
