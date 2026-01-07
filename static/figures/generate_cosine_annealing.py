#!/usr/bin/env python3
"""Generate cosine annealing learning rate schedule figure."""

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.size'] = 12

fig, ax = plt.subplots(figsize=(8, 5))

# Parameters
total_steps = 100000
max_lr = 1e-3
min_lr = 1e-5

# Generate cosine schedule
steps = np.arange(total_steps)
lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + np.cos(np.pi * steps / total_steps))

# Plot
ax.plot(steps, lr, color='#2563eb', linewidth=2.5)
ax.fill_between(steps, lr, min_lr, alpha=0.2, color='#2563eb')

# Add annotations
ax.axhline(y=max_lr, color='gray', linestyle='--', alpha=0.5, label='Max LR')
ax.axhline(y=min_lr, color='gray', linestyle=':', alpha=0.5, label='Min LR')

# Mark key points
ax.scatter([0], [max_lr], s=80, color='#16a34a', zorder=5)
ax.scatter([total_steps-1], [min_lr], s=80, color='#dc2626', zorder=5)

ax.text(5000, max_lr * 1.1, 'Max LR', fontsize=10, color='#16a34a', fontweight='bold')
ax.text(total_steps * 0.8, min_lr * 2, 'Min LR (η_min)', fontsize=10, color='#dc2626', fontweight='bold')

# Labels
ax.set_xlabel('Training Steps', fontsize=12)
ax.set_ylabel('Learning Rate', fontsize=12)
ax.set_title('Cosine Annealing Schedule', fontsize=14, fontweight='bold')
ax.set_xlim(0, total_steps)
ax.set_ylim(0, max_lr * 1.2)
ax.grid(True, alpha=0.3)

# Format y-axis as scientific notation
ax.ticklabel_format(axis='y', style='scientific', scilimits=(0,0))

# Add annotation box
ax.text(total_steps * 0.55, max_lr * 0.7, 
        'Used by:\nGPT-3, LLaMA,\nmost modern LLMs', 
        fontsize=10, ha='center',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

plt.tight_layout()
plt.savefig('cosine_annealing_schedule.png', bbox_inches='tight', facecolor='white')
plt.close()

print("Generated cosine_annealing_schedule.png")
