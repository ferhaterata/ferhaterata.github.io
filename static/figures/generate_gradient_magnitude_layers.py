#!/usr/bin/env python3
"""Generate visualization showing gradient magnitude decay through layers for different activations."""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.size'] = 11

# Number of layers
layers = np.arange(1, 21)

# Gradient decay for different scenarios
# Sigmoid: max derivative is 0.25, typical is even smaller (~0.1)
sigmoid_best = 0.25 ** layers  # Best case: all neurons at σ=0.5
sigmoid_typical = 0.1 ** layers  # Typical case: neurons partially saturated

# ReLU: derivative is 1 for positive (assume ~60% neurons active)
relu = 1.0 ** layers  # Gradient stays constant!
relu_with_dead = 0.6 ** layers  # Some dead neurons

# Tanh: similar to sigmoid
tanh_typical = 0.15 ** layers

# Create figure
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: Log scale
ax1 = axes[0]
ax1.semilogy(layers, sigmoid_best, 'r-', linewidth=2, marker='o', markersize=5, 
             label='Sigmoid (best case: 0.25ⁿ)')
ax1.semilogy(layers, sigmoid_typical, 'r--', linewidth=2, marker='s', markersize=5, 
             label='Sigmoid (typical: 0.1ⁿ)')
ax1.semilogy(layers, tanh_typical, 'm:', linewidth=2, marker='^', markersize=5, 
             label='Tanh (typical: 0.15ⁿ)')
ax1.semilogy(layers, relu, 'g-', linewidth=3, marker='o', markersize=5, 
             label='ReLU (ideal: 1.0ⁿ = 1)')
ax1.semilogy(layers, relu_with_dead, 'g--', linewidth=2, marker='s', markersize=5, 
             label='ReLU (with dead neurons: 0.6ⁿ)')

# Add danger zone
ax1.axhline(y=1e-6, color='red', linestyle=':', alpha=0.7)
ax1.text(10.5, 2e-6, 'Vanishing gradient zone', color='red', fontsize=10, style='italic')

ax1.set_xlabel('Layer (from output to input)', fontsize=12)
ax1.set_ylabel('Relative Gradient Magnitude (log scale)', fontsize=12)
ax1.set_title('Gradient Magnitude Decay During Backprop\n(Log Scale)', fontsize=13, fontweight='bold')
ax1.legend(loc='lower left', fontsize=9)
ax1.grid(True, alpha=0.3, which='both')
ax1.set_xlim(1, 20)
ax1.set_ylim(1e-20, 10)

# Right: Linear scale for first 5 layers (to show the decay clearly)
ax2 = axes[1]
layers_short = np.arange(1, 8)

sigmoid_short = 0.25 ** layers_short
relu_short = 1.0 ** layers_short

bars_width = 0.35
x = np.arange(len(layers_short))

bars1 = ax2.bar(x - bars_width/2, sigmoid_short, bars_width, label='Sigmoid (best: 0.25ⁿ)', color='red', alpha=0.7)
bars2 = ax2.bar(x + bars_width/2, relu_short, bars_width, label='ReLU (1.0ⁿ = 1)', color='green', alpha=0.7)

# Add value labels
for bar, val in zip(bars1, sigmoid_short):
    if val > 0.001:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                 f'{val:.3f}', ha='center', va='bottom', fontsize=9)

ax2.set_xlabel('Layers back from output', fontsize=12)
ax2.set_ylabel('Relative Gradient Magnitude', fontsize=12)
ax2.set_title('Gradient Decay Comparison\n(First 7 Layers)', fontsize=13, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels([f'Layer {i}' for i in layers_short])
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3, axis='y')

# Add annotation
ax2.annotate('After 5 layers:\nSigmoid: 0.001 (0.1%)\nReLU: 1.0 (100%)', 
             xy=(4, 0.001), xytext=(5, 0.5),
             fontsize=10, ha='left',
             arrowprops=dict(arrowstyle='->', color='black'),
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# Add main explanation
fig.text(0.5, -0.02, 
         'Key Insight: Sigmoid\'s max gradient (0.25) causes exponential decay. After 10 layers: 0.25¹⁰ = 10⁻⁶\n'
         'ReLU maintains gradient = 1 for positive activations, enabling training of deep networks.',
         ha='center', fontsize=11, style='italic', 
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.subplots_adjust(bottom=0.18)
plt.savefig('gradient_magnitude_layers.png', bbox_inches='tight', facecolor='white')
plt.close()

print("Generated gradient_magnitude_layers.png")
