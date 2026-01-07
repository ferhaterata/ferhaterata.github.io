#!/usr/bin/env python3
"""Generate cross-entropy vs MSE loss comparison figure."""

import matplotlib.pyplot as plt
import numpy as np

# Set style
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.size'] = 11

COLORS = {
    'primary': '#2563eb',      # Blue
    'secondary': '#dc2626',    # Red  
    'tertiary': '#16a34a',     # Green
}

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# For true label y=1, predicted probability p
p = np.linspace(0.01, 0.99, 500)

# MSE loss: (1 - p)^2  when y=1
mse = (1 - p) ** 2

# Cross-entropy loss: -log(p) when y=1
ce = -np.log(p)

# ===== Panel 1: Loss comparison =====
ax = axes[0]
ax.plot(p, mse, color=COLORS['secondary'], linewidth=2.5, label='MSE = (1 - ŷ)²')
ax.plot(p, ce, color=COLORS['primary'], linewidth=2.5, label='Cross-Entropy = -log(ŷ)')

ax.set_xlabel('Predicted Probability ŷ (when true label = 1)', fontsize=12)
ax.set_ylabel('Loss', fontsize=12)
ax.set_title('Loss Functions for Classification\n(True Label y = 1)', fontsize=13, fontweight='bold')
ax.legend(fontsize=10, loc='upper right')
ax.set_xlim(0, 1)
ax.set_ylim(0, 5)
ax.grid(True, alpha=0.3)

# Mark key points
# When ŷ = 0.99 (correct, confident)
ax.scatter([0.99], [(1-0.99)**2], color=COLORS['secondary'], s=80, zorder=5)
ax.scatter([0.99], [-np.log(0.99)], color=COLORS['primary'], s=80, zorder=5)
ax.annotate('ŷ=0.99 (correct)\nMSE=0.0001, CE=0.01', 
            xy=(0.99, 0.15), xytext=(0.7, 0.8),
            fontsize=9, arrowprops=dict(arrowstyle='->', color='gray'),
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

# When ŷ = 0.5 (uncertain)
ax.scatter([0.5], [(1-0.5)**2], color=COLORS['secondary'], s=80, zorder=5)
ax.scatter([0.5], [-np.log(0.5)], color=COLORS['primary'], s=80, zorder=5)
ax.axvline(x=0.5, color='gray', linestyle=':', alpha=0.5)
ax.annotate('ŷ=0.5 (uncertain)\nMSE=0.25, CE=0.69', 
            xy=(0.5, 0.69), xytext=(0.55, 1.8),
            fontsize=9, arrowprops=dict(arrowstyle='->', color='gray'),
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# When ŷ = 0.01 (wrong, confident) - KEY INSIGHT
ax.scatter([0.01], [(1-0.01)**2], color=COLORS['secondary'], s=80, zorder=5)
ax.scatter([0.01], [-np.log(0.01)], color=COLORS['primary'], s=80, zorder=5)
ax.annotate('ŷ=0.01 (WRONG!)\nMSE=0.98, CE=4.6', 
            xy=(0.05, 4.0), xytext=(0.2, 4.0),
            fontsize=9, arrowprops=dict(arrowstyle='->', color='red', lw=2),
            bbox=dict(boxstyle='round', facecolor='lightsalmon', alpha=0.9))

# ===== Panel 2: Key insight explanation =====
ax = axes[1]

# Bar chart comparing loss at key prediction values
predictions = ['ŷ=0.99\n(Correct)', 'ŷ=0.5\n(Uncertain)', 'ŷ=0.01\n(WRONG!)']
mse_values = [(1-0.99)**2, (1-0.5)**2, (1-0.01)**2]
ce_values = [-np.log(0.99), -np.log(0.5), -np.log(0.01)]

x_pos = np.array([0, 1, 2])
width = 0.35

bars1 = ax.bar(x_pos - width/2, mse_values, width, label='MSE', color=COLORS['secondary'], alpha=0.8)
bars2 = ax.bar(x_pos + width/2, ce_values, width, label='Cross-Entropy', color=COLORS['primary'], alpha=0.8)

# Add value labels
for bar, val in zip(bars1, mse_values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
            f'{val:.2f}', ha='center', fontsize=9)
for bar, val in zip(bars2, ce_values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
            f'{val:.2f}', ha='center', fontsize=9)

ax.set_ylabel('Loss Value', fontsize=12)
ax.set_title('Key Insight: CE Penalizes\nConfident Wrong Predictions Much More!', fontsize=13, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(predictions, fontsize=10)
ax.legend(fontsize=10)
ax.set_ylim(0, 6)
ax.grid(True, alpha=0.3, axis='y')

# Highlight the key difference
ax.annotate('5x more penalty!', 
            xy=(2 + width/2, ce_values[2]), xytext=(2.5, 3.5),
            fontsize=11, fontweight='bold', color='red',
            arrowprops=dict(arrowstyle='->', color='red', lw=2),
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.9))

plt.suptitle('Why Use Cross-Entropy for Classification?', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('cross_entropy_vs_mse.png', bbox_inches='tight', facecolor='white')
plt.close()

print("Generated cross_entropy_vs_mse.png")
