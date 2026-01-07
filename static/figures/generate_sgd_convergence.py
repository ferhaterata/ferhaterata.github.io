"""
Generate SGD convergence visualization for linear regression.
Shows loss curve and parameter trajectory during training.
"""

import numpy as np
import matplotlib.pyplot as plt

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13

def sgd_linear_regression_with_history(X, y, lr=0.01, epochs=100):
    """SGD for linear regression with loss and parameter history."""
    w1, w2 = 0.0, 0.0  # Initialize weights
    n = len(X)
    
    loss_history = []
    w1_history = []
    w2_history = []
    
    for epoch in range(epochs):
        # Compute epoch loss
        predictions = w1 + w2 * X
        epoch_loss = np.mean((predictions - y) ** 2)
        loss_history.append(epoch_loss)
        w1_history.append(w1)
        w2_history.append(w2)
        
        # Shuffle data
        indices = np.random.permutation(n)
        
        for i in indices:
            # Prediction error for single point
            error = (w1 + w2 * X[i]) - y[i]
            
            # Update weights based on this ONE point
            w1 = w1 - lr * 2 * error
            w2 = w2 - lr * 2 * error * X[i]
    
    return w1, w2, loss_history, w1_history, w2_history

# Generate data
np.random.seed(42)
X = np.array([1, 2, 3, 4, 5], dtype=float)
y = np.array([2.1, 4.0, 5.8, 8.1, 9.9])  # Approximately y = 2x

# Run SGD
w1_final, w2_final, losses, w1s, w2s = sgd_linear_regression_with_history(X, y, lr=0.01, epochs=50)

# Create figure
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# Plot 1: Loss convergence
ax1 = axes[0]
ax1.plot(losses, 'b-', linewidth=2, label='MSE Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss (MSE)')
ax1.set_title('SGD Convergence: Loss vs Epoch')
ax1.set_yscale('log')
ax1.axhline(y=losses[-1], color='r', linestyle='--', alpha=0.5, label=f'Final: {losses[-1]:.4f}')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Parameter trajectory
ax2 = axes[1]
ax2.plot(w1s, w2s, 'b.-', markersize=4, linewidth=1, alpha=0.7)
ax2.scatter([w1s[0]], [w2s[0]], c='green', s=100, marker='o', zorder=5, label='Start (0, 0)')
ax2.scatter([w1s[-1]], [w2s[-1]], c='red', s=100, marker='*', zorder=5, label=f'End ({w1s[-1]:.2f}, {w2s[-1]:.2f})')
ax2.scatter([0], [2], c='gold', s=150, marker='X', zorder=5, label='True (0, 2)')
ax2.set_xlabel('$w_1$ (intercept)')
ax2.set_ylabel('$w_2$ (slope)')
ax2.set_title('Parameter Trajectory')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Data and fitted line
ax3 = axes[2]
ax3.scatter(X, y, c='blue', s=80, label='Training data', zorder=5)
x_line = np.linspace(0, 6, 100)
y_true = 2 * x_line  # True line y = 2x
y_fitted = w1_final + w2_final * x_line
ax3.plot(x_line, y_true, 'g--', linewidth=2, label='True: y = 2x', alpha=0.7)
ax3.plot(x_line, y_fitted, 'r-', linewidth=2, label=f'Fitted: y = {w1_final:.2f} + {w2_final:.2f}x')
ax3.set_xlabel('x')
ax3.set_ylabel('y')
ax3.set_title('Linear Regression Fit')
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.set_xlim(0, 6)
ax3.set_ylim(0, 12)

plt.tight_layout()
plt.savefig('sgd_convergence.png', dpi=150, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.close()

print("Generated sgd_convergence.png")
print(f"Final weights: w1={w1_final:.4f}, w2={w2_final:.4f}")
print(f"Final loss: {losses[-1]:.6f}")
