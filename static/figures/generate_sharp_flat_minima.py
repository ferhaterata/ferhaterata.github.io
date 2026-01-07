#!/usr/bin/env python3
"""Generate sharp vs flat minima comparison figure."""

import matplotlib.pyplot as plt
import numpy as np

# Create figure with two subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Generate x values
x = np.linspace(-2, 2, 500)

# SAME perturbation for both
perturb = 0.5

# ===== Sharp minimum (narrow valley) =====
sharp = 2 * x**2 + 1  # Steeper parabola
loss_at_min_sharp = 1.0
loss_at_perturb_sharp = 2 * perturb**2 + 1  # = 1.5

ax1.plot(x, sharp, 'b-', linewidth=2.5)
ax1.plot(0, loss_at_min_sharp, 'ro', markersize=12, zorder=5, label='Minimum')

# Show perturbed points
ax1.plot([-perturb, perturb], [loss_at_perturb_sharp, loss_at_perturb_sharp], 'o', 
         color='orange', markersize=8, zorder=5)

# Horizontal dashed lines to show loss values
ax1.axhline(y=loss_at_min_sharp, color='red', linestyle='--', alpha=0.5, xmin=0.3, xmax=0.7)
ax1.axhline(y=loss_at_perturb_sharp, color='orange', linestyle='--', alpha=0.5, xmin=0.25, xmax=0.75)

# Vertical bar showing Δloss
ax1.annotate('', xy=(1.3, loss_at_perturb_sharp), xytext=(1.3, loss_at_min_sharp),
            arrowprops=dict(arrowstyle='<->', color='red', lw=2))
ax1.text(1.5, (loss_at_min_sharp + loss_at_perturb_sharp)/2, 
         f'Δloss = {loss_at_perturb_sharp - loss_at_min_sharp:.1f}', 
         fontsize=11, color='red', fontweight='bold', va='center')

# Label perturbation on x-axis
ax1.annotate('', xy=(perturb, 0.7), xytext=(0, 0.7),
            arrowprops=dict(arrowstyle='<->', color='gray', lw=1.5))
ax1.text(perturb/2, 0.55, f'Δw = {perturb}', fontsize=10, color='gray', ha='center')

ax1.set_xlabel('Parameter $w$', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12)
ax1.set_title('Sharp Minimum\n(Overfits)', fontsize=14, fontweight='bold', color='red')
ax1.set_xlim(-2, 2)
ax1.set_ylim(0.3, 3)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.grid(True, alpha=0.3)

# ===== Flat minimum (wide valley) =====
flat = 0.3 * x**2 + 1  # Shallower parabola
loss_at_min_flat = 1.0
loss_at_perturb_flat = 0.3 * perturb**2 + 1  # = 1.075

ax2.plot(x, flat, 'b-', linewidth=2.5)
ax2.plot(0, loss_at_min_flat, 'ro', markersize=12, zorder=5, label='Minimum')

# Show perturbed points (same perturbation!)
ax2.plot([-perturb, perturb], [loss_at_perturb_flat, loss_at_perturb_flat], 'o', 
         color='orange', markersize=8, zorder=5)

# Horizontal dashed lines
ax2.axhline(y=loss_at_min_flat, color='red', linestyle='--', alpha=0.5, xmin=0.3, xmax=0.7)
ax2.axhline(y=loss_at_perturb_flat, color='orange', linestyle='--', alpha=0.5, xmin=0.25, xmax=0.75)

# Vertical bar showing Δloss (much smaller!)
ax2.annotate('', xy=(1.3, loss_at_perturb_flat), xytext=(1.3, loss_at_min_flat),
            arrowprops=dict(arrowstyle='<->', color='green', lw=2))
ax2.text(1.5, (loss_at_min_flat + loss_at_perturb_flat)/2 + 0.05, 
         f'Δloss = {loss_at_perturb_flat - loss_at_min_flat:.2f}', 
         fontsize=11, color='green', fontweight='bold', va='center')

# Label perturbation on x-axis (same as sharp!)
ax2.annotate('', xy=(perturb, 0.7), xytext=(0, 0.7),
            arrowprops=dict(arrowstyle='<->', color='gray', lw=1.5))
ax2.text(perturb/2, 0.55, f'Δw = {perturb}', fontsize=10, color='gray', ha='center')

ax2.set_xlabel('Parameter $w$', fontsize=12)
ax2.set_ylabel('Loss', fontsize=12)
ax2.set_title('Flat Minimum\n(Generalizes)', fontsize=14, fontweight='bold', color='green')
ax2.set_xlim(-2, 2)
ax2.set_ylim(0.3, 3)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.grid(True, alpha=0.3)

# Add summary text at bottom
fig.text(0.5, 0.02, 
         'Same perturbation Δw = 0.5  →  Sharp: Δloss = 0.5 (big!)  |  Flat: Δloss = 0.08 (small)', 
         ha='center', fontsize=11, style='italic')

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig('sharp_flat_minima.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

print("Generated sharp_flat_minima.png")
