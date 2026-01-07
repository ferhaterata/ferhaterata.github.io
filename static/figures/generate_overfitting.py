#!/usr/bin/env python3
"""Generate overfitting vs good fit visualization."""

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.size'] = 12

# Create figure with three subplots
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4))

# Generate training data (noisy sine wave)
np.random.seed(42)
x_data = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
y_true = np.sin(x_data * 0.8)
y_data = y_true + np.random.normal(0, 0.15, len(x_data))

# Smooth x for plotting curves
x_smooth = np.linspace(0, 6, 200)

# ===== Panel 1: Training Data =====
ax1.scatter(x_data, y_data, s=100, c='#2563eb', marker='*', edgecolors='black', 
            linewidths=1, zorder=5, label='Training data')
ax1.set_xlim(0, 6)
ax1.set_ylim(-1.5, 1.5)
ax1.set_xlabel('x', fontsize=12)
ax1.set_ylabel('y', fontsize=12)
ax1.set_title('Training Data', fontsize=14, fontweight='bold')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.grid(True, alpha=0.3)

# ===== Panel 2: Overfitting (high-degree polynomial) =====
# Fit a high-degree polynomial (overfits)
coeffs_overfit = np.polyfit(x_data, y_data, deg=5)
y_overfit = np.polyval(coeffs_overfit, x_smooth)

ax2.scatter(x_data, y_data, s=100, c='#2563eb', marker='*', edgecolors='black', 
            linewidths=1, zorder=5)
ax2.plot(x_smooth, y_overfit, color='#dc2626', linewidth=2.5, label='Model')
ax2.set_xlim(0, 6)
ax2.set_ylim(-1.5, 1.5)
ax2.set_xlabel('x', fontsize=12)
ax2.set_ylabel('y', fontsize=12)
ax2.set_title('Overfitting\n(wiggly)', fontsize=14, fontweight='bold', color='#dc2626')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.grid(True, alpha=0.3)

# Add annotation
ax2.text(3, -1.2, 'Fits noise, not pattern', fontsize=10, ha='center', 
         style='italic', color='#dc2626')

# ===== Panel 3: Good Fit (appropriate complexity) =====
# Fit a lower-degree polynomial (good fit)
coeffs_good = np.polyfit(x_data, y_data, deg=2)
y_good = np.polyval(coeffs_good, x_smooth)

ax3.scatter(x_data, y_data, s=100, c='#2563eb', marker='*', edgecolors='black', 
            linewidths=1, zorder=5)
ax3.plot(x_smooth, y_good, color='#16a34a', linewidth=2.5, label='Model')
ax3.set_xlim(0, 6)
ax3.set_ylim(-1.5, 1.5)
ax3.set_xlabel('x', fontsize=12)
ax3.set_ylabel('y', fontsize=12)
ax3.set_title('Good Fit\n(smooth)', fontsize=14, fontweight='bold', color='#16a34a')
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.grid(True, alpha=0.3)

# Add annotation
ax3.text(3, -1.2, 'Captures pattern, ignores noise', fontsize=10, ha='center', 
         style='italic', color='#16a34a')

plt.tight_layout()
plt.savefig('overfitting_good_fit.png', bbox_inches='tight', facecolor='white')
plt.close()

print("Generated overfitting_good_fit.png")
