#!/usr/bin/env python3
"""Generate ill-conditioned loss landscape with SGD oscillations."""

import matplotlib.pyplot as plt
import numpy as np

# Set style
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.size'] = 11

COLORS = {
    'primary': '#2563eb',      # Blue
    'secondary': '#dc2626',    # Red  
    'tertiary': '#16a34a',     # Green
    'purple': '#9333ea',
}

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Ill-conditioned loss function: L(x,y) = 50*x^2 + y^2
# Condition number = 50 (ratio of largest to smallest eigenvalue)
# Steep direction: x (high curvature)
# Gentle direction: y (low curvature)

# Create meshgrid for contour plot
x = np.linspace(-1.5, 1.5, 200)
y = np.linspace(-1.5, 1.5, 200)
X, Y = np.meshgrid(x, y)

# Loss function (elongated ellipse)
kappa = 25  # Condition number
L = kappa * X**2 + Y**2

# ===== Panel 1: SGD oscillations =====
ax = axes[0]

# Draw contours
contours = ax.contour(X, Y, L, levels=15, cmap='Blues', linewidths=1.5, alpha=0.7)
ax.contourf(X, Y, L, levels=15, cmap='Blues', alpha=0.3)

# Simulate SGD with small learning rate
def sgd_step(x, y, lr):
    """Gradient: dL/dx = 2*kappa*x, dL/dy = 2*y"""
    grad_x = 2 * kappa * x
    grad_y = 2 * y
    return x - lr * grad_x, y - lr * grad_y

# SGD path (shows oscillations)
lr = 0.03
sgd_x, sgd_y = [1.2], [-1.2]
for _ in range(50):
    new_x, new_y = sgd_step(sgd_x[-1], sgd_y[-1], lr)
    sgd_x.append(new_x)
    sgd_y.append(new_y)

ax.plot(sgd_x, sgd_y, 'o-', color=COLORS['secondary'], markersize=3, linewidth=1.5, 
        label='SGD path (oscillates!)', alpha=0.8)
ax.scatter([sgd_x[0]], [sgd_y[0]], color=COLORS['secondary'], s=120, marker='*', 
           zorder=5, label='Start', edgecolors='black')
ax.scatter([0], [0], color=COLORS['tertiary'], s=150, marker='*', 
           zorder=5, label='Goal (minimum)', edgecolors='black')

# Annotate steep and gentle directions
ax.annotate('', xy=(0, 1.3), xytext=(0, -1.3),
            arrowprops=dict(arrowstyle='<->', color='gray', lw=2))
ax.text(0.15, 0, 'Gentle direction\n(slow progress)', fontsize=9, 
        color='gray', va='center', rotation=90)

ax.annotate('', xy=(0.8, 0), xytext=(-0.8, 0),
            arrowprops=dict(arrowstyle='<->', color='red', lw=2))
ax.text(0, 0.2, 'Steep direction\n(oscillates!)', fontsize=9, 
        color='red', ha='center')

ax.set_xlabel('$w_1$ (steep direction)', fontsize=12)
ax.set_ylabel('$w_2$ (gentle direction)', fontsize=12)
ax.set_title('Vanilla SGD: Oscillations in Ill-Conditioned Landscape\n(Condition number κ = 25)', 
             fontsize=12, fontweight='bold')
ax.legend(loc='upper right', fontsize=9)
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)

# ===== Panel 2: SGD vs Momentum comparison =====
ax = axes[1]

# Draw contours
contours = ax.contour(X, Y, L, levels=15, cmap='Blues', linewidths=1.5, alpha=0.7)
ax.contourf(X, Y, L, levels=15, cmap='Blues', alpha=0.3)

# SGD path (from before, truncated)
ax.plot(sgd_x[:30], sgd_y[:30], 'o-', color=COLORS['secondary'], markersize=3, 
        linewidth=1.5, label='SGD (oscillates)', alpha=0.7)

# Simulate momentum SGD
def momentum_step(x, y, vx, vy, lr, beta=0.9):
    """SGD with momentum"""
    grad_x = 2 * kappa * x
    grad_y = 2 * y
    vx_new = beta * vx + grad_x
    vy_new = beta * vy + grad_y
    return x - lr * vx_new, y - lr * vy_new, vx_new, vy_new

# Momentum path (much smoother!)
lr_mom = 0.01
mom_x, mom_y = [1.2], [-1.2]
vx, vy = 0, 0
for _ in range(50):
    new_x, new_y, vx, vy = momentum_step(mom_x[-1], mom_y[-1], vx, vy, lr_mom, beta=0.9)
    mom_x.append(new_x)
    mom_y.append(new_y)

ax.plot(mom_x[:30], mom_y[:30], 's-', color=COLORS['tertiary'], markersize=3, 
        linewidth=2, label='Momentum (smooth!)', alpha=0.9)

ax.scatter([1.2], [-1.2], color='black', s=120, marker='*', 
           zorder=5, label='Start', edgecolors='white')
ax.scatter([0], [0], color=COLORS['purple'], s=150, marker='*', 
           zorder=5, label='Goal', edgecolors='black')

# Add explanation
ax.text(0.02, 0.98, 
        'Why momentum helps:\n'
        '• Dampens oscillations in steep direction\n'
        '• Accumulates velocity in gentle direction\n'
        '• Faster convergence overall!',
        transform=ax.transAxes, fontsize=9, va='top',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

ax.set_xlabel('$w_1$ (steep direction)', fontsize=12)
ax.set_ylabel('$w_2$ (gentle direction)', fontsize=12)
ax.set_title('SGD vs Momentum: Why Momentum Helps', fontsize=12, fontweight='bold')
ax.legend(loc='lower right', fontsize=9)
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)

plt.suptitle('The Problem: Oscillations in Ill-Conditioned Landscapes', 
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('ill_conditioned_landscape.png', bbox_inches='tight', facecolor='white')
plt.close()

print("Generated ill_conditioned_landscape.png")
