#!/usr/bin/env python3
"""Generate XOR transformation visualization showing how hidden layers make XOR separable."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.size'] = 11

# XOR data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])  # XOR labels

# Transform through hidden layer (using weights from document)
# h1 = ReLU(x1 + x2 - 0.5) "at least one 1"
# h2 = ReLU(x1 + x2 - 1.5) "both are 1"
def hidden_layer(x):
    h1 = np.maximum(0, x[:, 0] + x[:, 1] - 0.5)
    h2 = np.maximum(0, x[:, 0] + x[:, 1] - 1.5)
    return np.column_stack([h1, h2])

H = hidden_layer(X)

# Create figure
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Left: Original XOR space (not separable)
ax1 = axes[0]
colors = ['red' if yi == 0 else 'blue' for yi in y]
ax1.scatter(X[:, 0], X[:, 1], c=colors, s=300, edgecolors='black', linewidth=2, zorder=5)

# Add labels
for i, (xi, yi, label) in enumerate(zip(X[:, 0], X[:, 1], y)):
    ax1.annotate(f'({xi:.0f},{yi:.0f})\ny={label}', xy=(xi, yi), 
                 xytext=(xi + 0.15, yi + 0.1), fontsize=10)

# Try to draw a line (impossible!)
ax1.axline((0, 0.5), slope=1, color='gray', linestyle='--', alpha=0.5, label='Any line fails!')
ax1.axline((0.5, 0), slope=-1, color='gray', linestyle=':', alpha=0.5)

ax1.set_xlim(-0.5, 1.8)
ax1.set_ylim(-0.5, 1.5)
ax1.set_xlabel('$x_1$', fontsize=14)
ax1.set_ylabel('$x_2$', fontsize=14)
ax1.set_title('Original Space\n(NOT Linearly Separable!)', fontsize=13, fontweight='bold', color='red')
ax1.set_aspect('equal')
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper right')

# Middle: Network diagram
ax2 = axes[1]
ax2.axis('off')
ax2.set_xlim(0, 10)
ax2.set_ylim(-2, 12)  # Expanded to accommodate text boxes

# Draw neurons
# Input layer
ax2.add_patch(plt.Circle((2, 7), 0.4, color='lightblue', ec='black', lw=2))
ax2.text(2, 7, '$x_1$', ha='center', va='center', fontsize=12, fontweight='bold')
ax2.add_patch(plt.Circle((2, 3), 0.4, color='lightblue', ec='black', lw=2))
ax2.text(2, 3, '$x_2$', ha='center', va='center', fontsize=12, fontweight='bold')

# Hidden layer
ax2.add_patch(plt.Circle((5, 8), 0.5, color='lightgreen', ec='black', lw=2))
ax2.text(5, 8, '$h_1$', ha='center', va='center', fontsize=12, fontweight='bold')
ax2.add_patch(plt.Circle((5, 2), 0.5, color='lightgreen', ec='black', lw=2))
ax2.text(5, 2, '$h_2$', ha='center', va='center', fontsize=12, fontweight='bold')

# Output
ax2.add_patch(plt.Circle((8, 5), 0.5, color='lightyellow', ec='black', lw=2))
ax2.text(8, 5, '$y$', ha='center', va='center', fontsize=12, fontweight='bold')

# Connections
for x1, y1, x2, y2 in [(2.4, 7, 4.5, 8), (2.4, 3, 4.5, 8), 
                       (2.4, 7, 4.5, 2), (2.4, 3, 4.5, 2),
                       (5.5, 8, 7.5, 5.2), (5.5, 2, 7.5, 4.8)]:
    ax2.annotate('', xy=(x2, y2), xytext=(x1, y1),
                 arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

# Add equations - positioned to avoid overlapping with neurons
ax2.text(5, 11, '$h_1 = \\mathrm{ReLU}(x_1 + x_2 - 0.5)$\n"At least one input is 1"', 
         ha='center', va='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
ax2.text(5, -1, '$h_2 = \\mathrm{ReLU}(x_1 + x_2 - 1.5)$\n"Both inputs are 1"', 
         ha='center', va='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

ax2.set_title('Hidden Layer\nCreates New Features', fontsize=13, fontweight='bold')

# Right: Transformed space (linearly separable!)
ax3 = axes[2]
colors = ['red' if yi == 0 else 'blue' for yi in y]
ax3.scatter(H[:, 0], H[:, 1], c=colors, s=300, edgecolors='black', linewidth=2, zorder=5)

# Add labels showing original inputs
for i, (h1, h2, x_orig, label) in enumerate(zip(H[:, 0], H[:, 1], X, y)):
    ax3.annotate(f'({x_orig[0]},{x_orig[1]})\ny={label}', xy=(h1, h2), 
                 xytext=(h1 + 0.1, h2 + 0.05), fontsize=10)

# Draw separating line (now possible!)
# Line equation: h2 = 0.4 * h1 - 0.15
# This correctly separates: RED (0,0) and (1.5,0.5) ABOVE the line
#                          BLUE (0.5,0) BELOW the line
# Verify: (0,0): 0 > -0.15 ✓ ABOVE
#         (0.5,0): 0 < 0.05 ✓ BELOW  
#         (1.5,0.5): 0.5 > 0.45 ✓ ABOVE
x_line = np.array([-0.2, 1.9])
y_line = 0.4 * x_line - 0.15
ax3.plot(x_line, y_line, 'g-', linewidth=4, label='Separating line!')
ax3.fill_between(x_line, y_line, 1.0, alpha=0.25, color='red')  # Above line = RED (y=0)
ax3.fill_between(x_line, y_line, -0.3, alpha=0.25, color='blue')  # Below line = BLUE (y=1)

ax3.set_xlim(-0.3, 1.8)
ax3.set_ylim(-0.3, 1.0)
ax3.set_xlabel('$h_1$ (at least one 1)', fontsize=12)
ax3.set_ylabel('$h_2$ (both are 1)', fontsize=12)
ax3.set_title('Transformed Space\n(LINEARLY Separable!)', fontsize=13, fontweight='bold', color='green')
ax3.set_aspect('equal')
ax3.grid(True, alpha=0.3)
ax3.legend(loc='upper right')

# Add main title
fig.suptitle('The Power of Hidden Layers: Transforming XOR to Become Linearly Separable', 
             fontsize=14, fontweight='bold', y=1.02)

# Add explanation
fig.text(0.5, -0.02, 
         'Key Insight: The hidden layer creates a NEW representation where the problem becomes easy!\n'
         'This is why deep learning works — each layer transforms data into more useful representations.',
         ha='center', fontsize=11, style='italic', 
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig('xor_transformation.png', bbox_inches='tight', facecolor='white')
plt.close()

print("Generated xor_transformation.png")
