#!/usr/bin/env python3
"""Generate bias-variance tradeoff / complexity tradeoff visualization."""

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.size'] = 12

# Create figure
fig, ax = plt.subplots(figsize=(10, 6))

# Generate complexity axis
complexity = np.linspace(0.5, 10, 200)

# Training error: decreases with complexity
train_error = 0.8 * np.exp(-0.5 * complexity) + 0.05

# Test error: U-shaped (decreases then increases)
test_error = 0.6 * np.exp(-0.4 * complexity) + 0.02 * (complexity - 3) ** 2 + 0.1

# Bias and variance curves (optional conceptual)
bias_squared = 0.8 * np.exp(-0.5 * complexity) + 0.02
variance = 0.005 * complexity ** 2 + 0.01

# Find optimal complexity
optimal_idx = np.argmin(test_error)
optimal_complexity = complexity[optimal_idx]
optimal_error = test_error[optimal_idx]

# Plot main curves
ax.plot(complexity, train_error, color='#2563eb', linewidth=2.5, label='Training Error')
ax.plot(complexity, test_error, color='#dc2626', linewidth=2.5, label='Test Error')

# Plot bias and variance (lighter, dashed)
ax.plot(complexity, bias_squared, color='#16a34a', linewidth=1.5, linestyle='--', alpha=0.7, label='Bias²')
ax.plot(complexity, variance, color='#f59e0b', linewidth=1.5, linestyle='--', alpha=0.7, label='Variance')

# Mark optimal point
ax.axvline(x=optimal_complexity, color='gray', linestyle=':', alpha=0.7)
ax.scatter([optimal_complexity], [optimal_error], s=150, color='#16a34a', zorder=5, edgecolors='white', linewidths=2)

# Shade regions
ax.axvspan(0.5, 2.5, alpha=0.1, color='#2563eb', label='_nolegend_')
ax.axvspan(6, 10, alpha=0.1, color='#dc2626', label='_nolegend_')

# Annotations
ax.annotate('Underfitting\n(High Bias)', xy=(1.5, 0.55), fontsize=11, ha='center', 
            color='#2563eb', fontweight='bold')
ax.annotate('Overfitting\n(High Variance)', xy=(8.5, 0.45), fontsize=11, ha='center',
            color='#dc2626', fontweight='bold')
ax.annotate('Sweet Spot\n(Optimal)', xy=(optimal_complexity, optimal_error - 0.08), fontsize=11, ha='center',
            color='#16a34a', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#16a34a', lw=1.5))

# Labels and title
ax.set_xlabel('Model Complexity', fontsize=13)
ax.set_ylabel('Error', fontsize=13)
ax.set_title('Bias-Variance Tradeoff', fontsize=16, fontweight='bold')

# Clean up axes
ax.set_xlim(0.5, 10)
ax.set_ylim(0, 0.7)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
ax.grid(True, alpha=0.3)

# Add annotation arrows for key points
ax.annotate('', xy=(2.5, 0.02), xytext=(1, 0.02),
            arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
ax.text(1.7, 0.04, 'More complex', fontsize=9, color='gray')

plt.tight_layout()
plt.savefig('complexity_tradeoff.png', bbox_inches='tight', facecolor='white')
plt.close()

print("Generated complexity_tradeoff.png")
