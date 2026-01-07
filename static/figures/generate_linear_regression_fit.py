#!/usr/bin/env python3
"""Generate linear regression fit visualization showing data, fitted line, and residuals."""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.size'] = 11

# Data from the document
X = np.array([1000, 1500, 2000, 2500, 3000])
y = np.array([200, 280, 350, 400, 500])

# Best fit parameters (from document)
w_good = 0.15
b_good = 50

# Bad fit parameters
w_bad = 0.05
b_bad = 150

# Predictions
y_pred_good = w_good * X + b_good
y_pred_bad = w_bad * X + b_bad

# Create figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left plot: Bad fit with residuals
ax1 = axes[0]
ax1.scatter(X, y, s=100, c='#1f77b4', edgecolors='black', zorder=5, label='Data points')
ax1.plot(X, y_pred_bad, 'r-', linewidth=2, label=f'Bad fit: y = {w_bad}x + {b_bad}')

# Draw residuals (vertical lines)
for xi, yi, y_predi in zip(X, y, y_pred_bad):
    ax1.plot([xi, xi], [yi, y_predi], 'r--', alpha=0.5, linewidth=1.5)
    ax1.annotate(f'{yi - y_predi:.0f}', xy=(xi + 50, (yi + y_predi) / 2), 
                 fontsize=9, color='red', ha='left')

ax1.set_xlabel('Size (sq ft)', fontsize=12)
ax1.set_ylabel('Price ($1000s)', fontsize=12)
ax1.set_title('Underfitting: Line Too Flat\nMSE = 13,730', fontsize=12, fontweight='bold')
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(800, 3200)
ax1.set_ylim(100, 550)

# Right plot: Good fit with residuals
ax2 = axes[1]
ax2.scatter(X, y, s=100, c='#1f77b4', edgecolors='black', zorder=5, label='Data points')
ax2.plot(X, y_pred_good, 'g-', linewidth=2, label=f'Good fit: y = {w_good}x + {b_good}')

# Draw residuals (vertical lines)
for xi, yi, y_predi in zip(X, y, y_pred_good):
    ax2.plot([xi, xi], [yi, y_predi], 'g--', alpha=0.5, linewidth=1.5)
    error = yi - y_predi
    if abs(error) > 1:  # Only annotate if error is visible
        ax2.annotate(f'{error:.0f}', xy=(xi + 50, (yi + y_predi) / 2), 
                     fontsize=9, color='green', ha='left')

ax2.set_xlabel('Size (sq ft)', fontsize=12)
ax2.set_ylabel('Price ($1000s)', fontsize=12)
ax2.set_title('Good Fit: Minimized Residuals\nMSE = 130', fontsize=12, fontweight='bold')
ax2.legend(loc='upper left')
ax2.grid(True, alpha=0.3)
ax2.set_xlim(800, 3200)
ax2.set_ylim(100, 550)

# Add explanation at bottom
fig.text(0.5, 0.02, 
         'Residuals (dashed lines) = Actual - Predicted. MSE squares and averages them.\n'
         'Goal: Find w and b that minimize total squared residuals.',
         ha='center', fontsize=10, style='italic', 
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.subplots_adjust(bottom=0.15)
plt.savefig('linear_regression_fit.png', bbox_inches='tight', facecolor='white')
plt.close()

print("Generated linear_regression_fit.png")
