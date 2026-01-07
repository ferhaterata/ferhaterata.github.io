#!/usr/bin/env python3
"""Generate a simple 1D loss landscape diagram."""

import matplotlib.pyplot as plt
import numpy as np

# Create figure
fig, ax = plt.subplots(1, 1, figsize=(8, 5))

# Generate smooth loss curve
w = np.linspace(-3, 3, 500)
# Create a smooth valley shape
loss = 0.5 * (w ** 2) + 0.3 * np.exp(-2 * (w - 0.2)**2) * np.cos(3 * w) + 1

# Plot loss curve
ax.plot(w, loss, 'b-', linewidth=2.5, label='Loss $\\ell(w)$')

# Find minimum
min_idx = np.argmin(loss)
min_w = w[min_idx]
min_loss = loss[min_idx]

# Mark minimum
ax.plot(min_w, min_loss, 'ro', markersize=10, zorder=5)
ax.annotate('Minimum\n(goal)', 
            xy=(min_w, min_loss), 
            xytext=(min_w + 1, min_loss + 0.5),
            fontsize=11,
            arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
            color='red')

# Mark current position
current_w = -1.5
current_loss = 0.5 * (current_w ** 2) + 0.3 * np.exp(-2 * (current_w - 0.2)**2) * np.cos(3 * current_w) + 1
ax.plot(current_w, current_loss, 'go', markersize=10, zorder=5)
ax.annotate('Current $w_t$', 
            xy=(current_w, current_loss), 
            xytext=(current_w - 0.8, current_loss + 0.6),
            fontsize=11,
            arrowprops=dict(arrowstyle='->', color='green', lw=1.5),
            color='green')

# Labels
ax.set_xlabel('Parameter $w$', fontsize=12)
ax.set_ylabel('Loss $\\ell(w)$', fontsize=12)
ax.set_title('Gradient Descent: Finding the Valley', fontsize=14, fontweight='bold')

# Style
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlim(-3, 3)
ax.set_ylim(0.5, 3)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('loss_valley.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

print("Generated loss_valley.png")
