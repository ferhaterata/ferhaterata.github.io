"""
Generate sigmoid function and derivative plot showing:
1. Sigmoid function σ(z)
2. Sigmoid derivative σ'(z) = σ(z)(1-σ(z))
3. Maximum derivative at z=0 where σ'(0) = 0.25
"""

import numpy as np
import matplotlib.pyplot as plt

# Set up the figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Generate x values
z = np.linspace(-6, 6, 500)

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Sigmoid derivative
def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)

# Calculate values
sigma = sigmoid(z)
sigma_prime = sigmoid_derivative(z)

# === Plot 1: Sigmoid Function ===
ax1.plot(z, sigma, 'b-', linewidth=2.5, label=r'$\sigma(z) = \frac{1}{1+e^{-z}}$')
ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='y = 0.5')
ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
ax1.axhline(y=1, color='gray', linestyle='-', alpha=0.3)
ax1.axvline(x=0, color='gray', linestyle='-', alpha=0.3)

# Mark the midpoint
ax1.plot(0, 0.5, 'ro', markersize=8, zorder=5)
ax1.annotate(r'$\sigma(0) = 0.5$', xy=(0, 0.5), xytext=(1.5, 0.6),
             fontsize=11, arrowprops=dict(arrowstyle='->', color='red', alpha=0.7))

ax1.set_xlabel('z (input)', fontsize=12)
ax1.set_ylabel(r'$\sigma(z)$', fontsize=12)
ax1.set_title('Sigmoid Function', fontsize=14, fontweight='bold')
ax1.set_xlim(-6, 6)
ax1.set_ylim(-0.1, 1.1)
ax1.legend(loc='lower right', fontsize=10)
ax1.grid(True, alpha=0.3)

# Add saturation regions annotation
ax1.annotate('Saturation\n(σ → 0)', xy=(-5, 0.01), fontsize=9, ha='center', 
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
ax1.annotate('Saturation\n(σ → 1)', xy=(5, 0.99), fontsize=9, ha='center',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# === Plot 2: Sigmoid Derivative ===
ax2.plot(z, sigma_prime, 'r-', linewidth=2.5, label=r"$\sigma'(z) = \sigma(z)(1-\sigma(z))$")
ax2.axhline(y=0.25, color='green', linestyle='--', alpha=0.7, linewidth=1.5, label='Maximum = 0.25')
ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
ax2.axvline(x=0, color='gray', linestyle='-', alpha=0.3)

# Mark the maximum point
ax2.plot(0, 0.25, 'go', markersize=10, zorder=5)
ax2.annotate(r"$\sigma'(0) = 0.5 \times 0.5 = \mathbf{0.25}$", 
             xy=(0, 0.25), xytext=(2, 0.22),
             fontsize=11, fontweight='bold',
             arrowprops=dict(arrowstyle='->', color='green', lw=1.5),
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

# Add text explaining the bound
ax2.text(0, 0.35, r"Maximum possible derivative $= \frac{1}{4}$",
         ha='center', fontsize=11, fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='white', edgecolor='green', alpha=0.9))

ax2.set_xlabel('z (input)', fontsize=12)
ax2.set_ylabel(r"$\sigma'(z)$", fontsize=12)
ax2.set_title("Sigmoid Derivative (Why Vanishing Gradients)", fontsize=14, fontweight='bold')
ax2.set_xlim(-6, 6)
ax2.set_ylim(-0.05, 0.4)
ax2.legend(loc='upper right', fontsize=10)
ax2.grid(True, alpha=0.3)

# Add annotation about vanishing gradients
ax2.annotate('In saturation regions:\n' + r"$\sigma' \to 0$" + '\n(vanishing gradient!)', 
             xy=(-4, 0.02), fontsize=9, ha='center',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig('sigmoid_derivative.png', dpi=150, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.close()

print("Generated: sigmoid_derivative.png")

# === Additional figure: Comparison with ReLU ===
fig2, ax3 = plt.subplots(figsize=(10, 5))

z_relu = np.linspace(-3, 3, 500)

# ReLU and its derivative
relu = np.maximum(0, z_relu)
relu_derivative = np.where(z_relu > 0, 1, 0)

# Plot sigmoid derivative for comparison
ax3.plot(z_relu, sigmoid_derivative(z_relu), 'r-', linewidth=2.5, 
         label=r"Sigmoid: $\sigma'(z) = \sigma(z)(1-\sigma(z))$ (max = 0.25)")
ax3.plot(z_relu, relu_derivative, 'b-', linewidth=2.5, alpha=0.8,
         label=r"ReLU: $\frac{d}{dz}\max(0,z) = \mathbf{1}$ for $z > 0$")

ax3.axhline(y=1, color='blue', linestyle='--', alpha=0.5)
ax3.axhline(y=0.25, color='red', linestyle='--', alpha=0.5)
ax3.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
ax3.axvline(x=0, color='gray', linestyle='-', alpha=0.3)

# Annotations
ax3.annotate('ReLU gradient = 1\n(no decay!)', xy=(2, 1), xytext=(2.5, 0.75),
             fontsize=11, arrowprops=dict(arrowstyle='->', color='blue'),
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

ax3.annotate('Sigmoid max = 0.25\n(gradient shrinks 4× per layer)', 
             xy=(0, 0.25), xytext=(-2, 0.5),
             fontsize=11, arrowprops=dict(arrowstyle='->', color='red'),
             bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))

ax3.set_xlabel('z (input)', fontsize=12)
ax3.set_ylabel('Derivative', fontsize=12)
ax3.set_title('Why ReLU Solves Vanishing Gradients', fontsize=14, fontweight='bold')
ax3.set_xlim(-3, 3)
ax3.set_ylim(-0.1, 1.2)
ax3.legend(loc='center right', fontsize=10)
ax3.grid(True, alpha=0.3)

# Add text box with key insight
textstr = 'After 10 layers:\n' + \
          r'Sigmoid: $0.25^{10} \approx 10^{-6}$ (vanishes!)' + '\n' + \
          r'ReLU: $1^{10} = 1$ (preserved!)'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.9)
ax3.text(0.02, 0.98, textstr, transform=ax3.transAxes, fontsize=10,
         verticalalignment='top', bbox=props)

plt.tight_layout()
plt.savefig('sigmoid_vs_relu_derivative.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()

print("Generated: sigmoid_vs_relu_derivative.png")
