#!/usr/bin/env python3
"""Generate step decay learning rate schedule figure."""

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.size'] = 12

fig, ax = plt.subplots(figsize=(8, 5))

# Parameters (100 epochs, steps at 30, 60, 90)
total_epochs = 100
milestones = [30, 60, 90]
initial_lr = 0.1
gamma = 0.1  # Divide by 10 at each milestone

# Generate LR schedule
epochs = np.arange(total_epochs + 1)
lr = np.ones_like(epochs, dtype=float) * initial_lr

for milestone in milestones:
    lr[epochs >= milestone] *= gamma

# Plot staircase
ax.step(epochs, lr, where='post', color='#2563eb', linewidth=2.5)

# Add milestone markers and annotations
for i, milestone in enumerate(milestones):
    lr_at_milestone = initial_lr * (gamma ** (i + 1))
    ax.axvline(x=milestone, color='gray', linestyle='--', alpha=0.5)
    ax.scatter([milestone], [lr_at_milestone], s=60, color='#dc2626', zorder=5)
    ax.text(milestone, lr_at_milestone * 1.5, f'Epoch {milestone}', 
            ha='center', fontsize=9, color='#dc2626')

# Labels
ax.set_xlabel('Epochs', fontsize=12)
ax.set_ylabel('Learning Rate', fontsize=12)
ax.set_title('Step Decay Schedule', fontsize=14, fontweight='bold')
ax.set_xlim(0, total_epochs)
ax.set_ylim(1e-5, 0.2)
ax.set_yscale('log')
ax.grid(True, alpha=0.3, which='both')

# Add annotation box
ax.text(75, 0.05, 'Divide by 10\nat epochs\n30, 60, 90', 
        fontsize=10, ha='center',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

plt.tight_layout()
plt.savefig('step_decay_schedule.png', bbox_inches='tight', facecolor='white')
plt.close()

print("Generated step_decay_schedule.png")
