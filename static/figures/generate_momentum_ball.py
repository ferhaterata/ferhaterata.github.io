#!/usr/bin/env python3
"""Generate momentum 'ball rolling downhill' intuition figure."""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches

plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.size'] = 12

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Colors
RED = '#dc2626'
GREEN = '#16a34a'
GRAY = '#6b7280'

# ===== Panel 1: Without momentum (oscillates) =====
ax = axes[0]
ax.set_xlim(-0.5, 5)
ax.set_ylim(-2, 2)
ax.set_aspect('equal')
ax.axis('off')

# Draw oscillating path
# Start at top left, oscillate down to bottom right
steps_no_mom = [
    (0, 1.5),
    (0.5, -0.8),
    (1.0, 1.2),
    (1.5, -0.5),
    (2.0, 0.9),
    (2.5, -0.2),
    (3.0, 0.6),
    (3.5, 0.1),
    (4.0, 0.3),
    (4.5, 0)  # goal
]

# Draw arrows between steps
for i in range(len(steps_no_mom) - 1):
    x1, y1 = steps_no_mom[i]
    x2, y2 = steps_no_mom[i + 1]
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=RED, lw=2))

# Mark start and goal
ax.scatter([0], [1.5], s=150, color=RED, marker='o', zorder=5, edgecolors='black')
ax.scatter([4.5], [0], s=200, color=GREEN, marker='*', zorder=5, edgecolors='black')
ax.text(0, 1.8, 'Start', ha='center', fontsize=10, fontweight='bold')
ax.text(4.5, -0.4, 'Goal', ha='center', fontsize=10, fontweight='bold', color=GREEN)

# Add label
ax.text(2.25, -1.8, '(oscillates back & forth)', ha='center', fontsize=11, style='italic', color=GRAY)
ax.set_title('Without Momentum', fontsize=14, fontweight='bold', color=RED, pad=10)

# ===== Panel 2: With momentum (smooth path) =====
ax = axes[1]
ax.set_xlim(-0.5, 5)
ax.set_ylim(-2, 2)
ax.set_aspect('equal')
ax.axis('off')

# Draw smooth path (almost straight to goal)
steps_momentum = [
    (0, 1.5),
    (0.6, 1.2),
    (1.2, 0.9),
    (1.8, 0.7),
    (2.4, 0.5),
    (3.0, 0.35),
    (3.6, 0.2),
    (4.2, 0.08),
    (4.5, 0)  # goal
]

# Draw arrows between steps
for i in range(len(steps_momentum) - 1):
    x1, y1 = steps_momentum[i]
    x2, y2 = steps_momentum[i + 1]
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=GREEN, lw=2.5))

# Mark start and goal
ax.scatter([0], [1.5], s=150, color=GREEN, marker='o', zorder=5, edgecolors='black')
ax.scatter([4.5], [0], s=200, color=GREEN, marker='*', zorder=5, edgecolors='black')
ax.text(0, 1.8, 'Start', ha='center', fontsize=10, fontweight='bold')
ax.text(4.5, -0.4, 'Goal', ha='center', fontsize=10, fontweight='bold', color=GREEN)

# Add label
ax.text(2.25, -1.8, '(smooth, direct path)', ha='center', fontsize=11, style='italic', color=GREEN)
ax.set_title('With Momentum', fontsize=14, fontweight='bold', color=GREEN, pad=10)

# Add explanation box at bottom
fig.text(0.5, 0.02, 
         '• Perpendicular oscillations cancel out (+g then -g averages to 0)\n'
         '• Consistent direction accumulates (gradients add up)\n'
         '• Result: Faster progress along the valley!',
         ha='center', fontsize=10,
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

plt.suptitle('Intuition: Ball Rolling Downhill', fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0.12, 1, 0.95])
plt.savefig('momentum_ball.png', bbox_inches='tight', facecolor='white')
plt.close()

print("Generated momentum_ball.png")
