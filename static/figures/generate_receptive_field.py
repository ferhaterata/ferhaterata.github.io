"""
Generate receptive field visualization for CNNs.
Shows how receptive field grows with depth.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle, FancyBboxPatch
import numpy as np

# Set up the figure
fig, axes = plt.subplots(1, 3, figsize=(14, 5))

def draw_grid(ax, size, highlight_cells=None, highlight_color='#FF5722', 
              grid_color='#E0E0E0', title='', receptive_field_size=None):
    """Draw a grid representing a feature map."""
    ax.set_xlim(-0.5, size - 0.5)
    ax.set_ylim(-0.5, size - 0.5)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Draw grid cells
    for i in range(size):
        for j in range(size):
            rect = Rectangle((j - 0.5, size - i - 1.5), 1, 1, 
                            facecolor='white', edgecolor=grid_color, linewidth=1)
            ax.add_patch(rect)
    
    # Highlight cells
    if highlight_cells:
        for (i, j) in highlight_cells:
            rect = Rectangle((j - 0.5, size - i - 1.5), 1, 1, 
                            facecolor=highlight_color, edgecolor='black', 
                            linewidth=2, alpha=0.7)
            ax.add_patch(rect)
    
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    
    if receptive_field_size:
        ax.text(size/2 - 0.5, -1.2, f'Receptive Field: {receptive_field_size}×{receptive_field_size}', 
                ha='center', fontsize=10, color='#424242')

# Input image (7x7)
ax1 = axes[0]
# Highlight 3x3 receptive field for layer 1
highlight1 = [(i, j) for i in range(2, 5) for j in range(2, 5)]
draw_grid(ax1, 7, highlight_cells=highlight1, highlight_color='#FFCDD2',
          title='Input Image (7×7)', receptive_field_size=3)

# After Conv Layer 1 (5x5)
ax2 = axes[1]
# Highlight 3x3 receptive field for layer 2
highlight2 = [(i, j) for i in range(1, 4) for j in range(1, 4)]
draw_grid(ax2, 5, highlight_cells=highlight2, highlight_color='#BBDEFB',
          title='After Conv1 (5×5)\n3×3 kernel, stride 1', receptive_field_size=5)

# After Conv Layer 2 (3x3)
ax3 = axes[2]
# Highlight center cell
highlight3 = [(1, 1)]
draw_grid(ax3, 3, highlight_cells=highlight3, highlight_color='#C8E6C9',
          title='After Conv2 (3×3)\n3×3 kernel, stride 1', receptive_field_size=7)

# Draw arrows between grids
arrow_props = dict(arrowstyle='->', color='#616161', lw=2)

# Arrow 1
fig.patches.append(mpatches.FancyArrowPatch(
    (0.35, 0.5), (0.38, 0.5),
    transform=fig.transFigure,
    arrowstyle='->', mutation_scale=15, color='#616161', lw=2
))

# Arrow 2
fig.patches.append(mpatches.FancyArrowPatch(
    (0.64, 0.5), (0.67, 0.5),
    transform=fig.transFigure,
    arrowstyle='->', mutation_scale=15, color='#616161', lw=2
))

# Add explanation at bottom
fig.text(0.5, 0.02, 
         'Each Conv layer with 3×3 kernel increases receptive field by 2 pixels per side.\n'
         'Layer 1 RF = 3×3 → Layer 2 RF = 5×5 → Layer 3 RF = 7×7 (sees entire input!)',
         ha='center', fontsize=11, color='#424242',
         bbox=dict(boxstyle='round', facecolor='#FFF9C4', edgecolor='#FFD54F', alpha=0.9))

plt.suptitle('Receptive Field Growth in CNNs', fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0.12, 1, 0.95])
plt.savefig('receptive_field.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()

print("Generated receptive_field.png")
