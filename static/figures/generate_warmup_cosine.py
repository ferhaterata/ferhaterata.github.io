#!/usr/bin/env python3
"""Generate warmup + cosine learning rate schedule figure (Standard LLM Recipe)."""

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.size'] = 12

fig, ax = plt.subplots(figsize=(10, 5))

# Parameters
warmup_steps = 2000
total_steps = 100000
max_lr = 1e-3
min_lr = 1e-5

# Generate combined schedule
steps = np.arange(total_steps)
lr = np.zeros_like(steps, dtype=float)

# Warmup phase (linear)
warmup_mask = steps < warmup_steps
lr[warmup_mask] = max_lr * steps[warmup_mask] / warmup_steps

# Cosine decay phase
cosine_mask = steps >= warmup_steps
cosine_steps = steps[cosine_mask] - warmup_steps
cosine_total = total_steps - warmup_steps
lr[cosine_mask] = min_lr + 0.5 * (max_lr - min_lr) * (1 + np.cos(np.pi * cosine_steps / cosine_total))

# Plot
ax.plot(steps, lr, color='#2563eb', linewidth=2.5)
ax.fill_between(steps, lr, min_lr, alpha=0.15, color='#2563eb')

# Add warmup region shading
ax.axvspan(0, warmup_steps, alpha=0.1, color='#16a34a')
ax.axvline(x=warmup_steps, color='#16a34a', linestyle='--', alpha=0.7, linewidth=1.5)

# Horizontal reference lines
ax.axhline(y=max_lr, color='gray', linestyle=':', alpha=0.5)
ax.axhline(y=min_lr, color='gray', linestyle=':', alpha=0.5)

# Mark key points
ax.scatter([0], [0], s=60, color='#16a34a', zorder=5)
ax.scatter([warmup_steps], [max_lr], s=80, color='#16a34a', zorder=5)
ax.scatter([total_steps-1], [min_lr], s=80, color='#dc2626', zorder=5)

# Annotations
ax.text(warmup_steps / 2, max_lr * 0.35, 'Warmup', ha='center', fontsize=11, 
        color='#16a34a', fontweight='bold')

ax.annotate('', xy=(warmup_steps, -max_lr * 0.08), xytext=(0, -max_lr * 0.08),
            arrowprops=dict(arrowstyle='<->', color='#16a34a', lw=1.5),
            annotation_clip=False)

ax.text(total_steps / 2, max_lr * 0.75, 'Cosine Decay', ha='center', fontsize=11,
        color='#2563eb', fontweight='bold')

ax.text(warmup_steps * 0.6, -max_lr * 0.18, 'warmup', ha='center', fontsize=9, 
        color='#16a34a')

ax.annotate('', xy=(total_steps, -max_lr * 0.08), xytext=(warmup_steps, -max_lr * 0.08),
            arrowprops=dict(arrowstyle='<->', color='#2563eb', lw=1.5),
            annotation_clip=False)

ax.text((warmup_steps + total_steps) / 2, -max_lr * 0.18, 'total T', ha='center', fontsize=9,
        color='#2563eb')

# Labels
ax.set_xlabel('Training Steps', fontsize=12)
ax.set_ylabel('Learning Rate', fontsize=12)
ax.set_title('Warmup + Cosine Schedule (Standard LLM Recipe)', fontsize=14, fontweight='bold')
ax.set_xlim(0, total_steps)
ax.set_ylim(-max_lr * 0.25, max_lr * 1.15)
ax.grid(True, alpha=0.3)

# Format y-axis
ax.ticklabel_format(axis='y', style='scientific', scilimits=(0,0))

# Add annotation box
ax.text(total_steps * 0.75, max_lr * 0.55, 
        'Used by:\nGPT-3, LLaMA,\nmost modern LLMs', 
        fontsize=10, ha='center',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

plt.tight_layout()
plt.savefig('warmup_cosine_schedule.png', bbox_inches='tight', facecolor='white')
plt.close()

print("Generated warmup_cosine_schedule.png")
