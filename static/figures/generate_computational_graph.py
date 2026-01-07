"""
Generate computational graph visualization for backpropagation explanation.
Shows forward pass (blue) and backward pass (red) with gradient flow.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Set up the figure
fig, ax = plt.subplots(1, 1, figsize=(14, 6))
ax.set_xlim(-0.5, 13.5)
ax.set_ylim(-2, 4)
ax.set_aspect('equal')
ax.axis('off')

# Colors
forward_color = '#2196F3'  # Blue
backward_color = '#F44336'  # Red
node_color = '#E3F2FD'
operation_color = '#FFF3E0'

# Node positions (x, y)
nodes = {
    'x': (0, 1),
    'W1': (1.5, 2.5),
    'mul1': (2, 1),
    'b1': (3, 2.5),
    'add1': (3.5, 1),
    'relu': (5, 1),
    'h': (6.5, 1),
    'W2': (7.5, 2.5),
    'mul2': (8, 1),
    'b2': (9, 2.5),
    'add2': (9.5, 1),
    'sigmoid': (11, 1),
    'y_hat': (12.5, 1),
}

# Draw input/output nodes (circles)
circle_nodes = ['x', 'W1', 'b1', 'h', 'W2', 'b2', 'y_hat']
for name in circle_nodes:
    x, y = nodes[name]
    circle = plt.Circle((x, y), 0.35, color=node_color, ec='black', linewidth=1.5, zorder=3)
    ax.add_patch(circle)
    ax.text(x, y, name.replace('_', '\n'), ha='center', va='center', fontsize=10, fontweight='bold', zorder=4)

# Draw operation nodes (rounded rectangles)
op_nodes = {'mul1': '×', 'add1': '+', 'relu': 'ReLU', 'mul2': '×', 'add2': '+', 'sigmoid': 'σ'}
for name, label in op_nodes.items():
    x, y = nodes[name]
    width = 0.8 if len(label) > 1 else 0.5
    rect = FancyBboxPatch((x - width/2, y - 0.3), width, 0.6, 
                          boxstyle="round,pad=0.05,rounding_size=0.15",
                          facecolor=operation_color, edgecolor='black', linewidth=1.5, zorder=3)
    ax.add_patch(rect)
    ax.text(x, y, label, ha='center', va='center', fontsize=10, fontweight='bold', zorder=4)

# Forward pass arrows (blue)
forward_edges = [
    ('x', 'mul1'),
    ('W1', 'mul1'),
    ('mul1', 'add1'),
    ('b1', 'add1'),
    ('add1', 'relu'),
    ('relu', 'h'),
    ('h', 'mul2'),
    ('W2', 'mul2'),
    ('mul2', 'add2'),
    ('b2', 'add2'),
    ('add2', 'sigmoid'),
    ('sigmoid', 'y_hat'),
]

def get_edge_points(start, end, offset=0.4):
    x1, y1 = nodes[start]
    x2, y2 = nodes[end]
    # Direction vector
    dx = x2 - x1
    dy = y2 - y1
    length = np.sqrt(dx**2 + dy**2)
    # Normalize
    dx /= length
    dy /= length
    # Offset from centers
    return (x1 + dx * offset, y1 + dy * offset, x2 - dx * offset, y2 - dy * offset)

for start, end in forward_edges:
    x1, y1, x2, y2 = get_edge_points(start, end)
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=forward_color, lw=2),
                zorder=2)

# Title for forward pass
ax.text(6.5, 3.5, 'Forward Pass', ha='center', va='center', fontsize=14, 
        fontweight='bold', color=forward_color)

# Draw backward gradient flow (red, below the main graph)
backward_y = -1

# Backward flow labels
backward_labels = [
    (12.5, '∂L/∂ŷ'),
    (11, '∂L/∂z₂'),
    (9.5, ''),
    (8, ''),
    (6.5, '∂L/∂h'),
    (5, '∂L/∂z₁'),
    (3.5, ''),
    (2, ''),
    (0, '∂L/∂x'),
]

# Draw backward arrows
for i in range(len(backward_labels) - 1):
    x1 = backward_labels[i][0]
    x2 = backward_labels[i + 1][0]
    ax.annotate('', xy=(x2 + 0.3, backward_y), xytext=(x1 - 0.3, backward_y),
                arrowprops=dict(arrowstyle='->', color=backward_color, lw=2),
                zorder=2)

# Add gradient labels
for x, label in backward_labels:
    if label:
        ax.text(x, backward_y - 0.5, label, ha='center', va='center', fontsize=9, 
                color=backward_color, fontweight='bold')

# Title for backward pass
ax.text(6.5, -1.7, 'Backward Pass (Gradient Flow)', ha='center', va='center', 
        fontsize=14, fontweight='bold', color=backward_color)

# Add vertical dashed lines connecting forward and backward
connect_points = [12.5, 11, 6.5, 5, 0]
for x in connect_points:
    ax.plot([x, x], [0.6, backward_y + 0.3], 'k--', alpha=0.3, lw=1)

# Add legend
legend_elements = [
    mpatches.Patch(facecolor=forward_color, label='Forward pass'),
    mpatches.Patch(facecolor=backward_color, label='Backward pass (gradients)'),
    mpatches.Patch(facecolor=node_color, edgecolor='black', label='Variables (data)'),
    mpatches.Patch(facecolor=operation_color, edgecolor='black', label='Operations'),
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

plt.tight_layout()
plt.savefig('computational_graph.png', dpi=150, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.close()

print("Generated computational_graph.png")
