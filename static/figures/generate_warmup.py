#!/usr/bin/env python3
"""Generate linear warmup learning rate schedule figure."""

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.size'] = 12

fig, ax = plt.subplots(figsize=(8, 5))

# Parameters
warmup_steps = 2000
total_steps = 10000
max_lr = 1e-3

# Generate LR schedule
steps = np.arange(total_steps)
lr = np.where(steps < warmup_steps,
              max_lr * steps / warmup_steps,  # Linear warmup
              max_lr)  # Constant after warmup

# Plot
ax.plot(steps, lr, color='#2563eb', linewidth=2.5)
ax.fill_between(steps[:warmup_steps+1], lr[:warmup_steps+1], alpha=0.3, color='#2563eb')

# Add annotations
ax.axvline(x=warmup_steps, color='gray', linestyle='--', alpha=0.7)
ax.axhline(y=max_lr, color='gray', linestyle=':', alpha=0.5)

# Label the full LR
ax.annotate('Full LR after warmup', 
            xy=(warmup_steps + 500, max_lr), 
            xytext=(warmup_steps + 1500, max_lr * 0.85),
            fontsize=11,
            arrowprops=dict(arrowstyle='->', color='gray'),
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

# Label warmup region
ax.text(warmup_steps / 2, max_lr * 0.3, 'Warmup\nregion', 
        ha='center', fontsize=11, style='italic', color='#2563eb')

# Mark warmup endpoint
ax.scatter([warmup_steps], [max_lr], s=80, color='#dc2626', zorder=5)
ax.text(warmup_steps, max_lr * 1.08, f'{warmup_steps} steps', 
        ha='center', fontsize=9, color='#dc2626')

# Labels
ax.set_xlabel('Training Steps', fontsize=12)
ax.set_ylabel('Learning Rate', fontsize=12)
ax.set_title('Linear Warmup Schedule', fontsize=14, fontweight='bold')
ax.set_xlim(0, total_steps)
ax.set_ylim(0, max_lr * 1.15)
ax.grid(True, alpha=0.3)

# Format y-axis as scientific notation
ax.ticklabel_format(axis='y', style='scientific', scilimits=(0,0))

plt.tight_layout()
plt.savefig('warmup_schedule.png', bbox_inches='tight', facecolor='white')
plt.close()

print("Generated warmup_schedule.png")
