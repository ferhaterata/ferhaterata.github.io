"""
Generate dropout ensemble visualization.
Shows how dropout creates implicit ensemble of networks.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle, FancyBboxPatch, FancyArrowPatch
import numpy as np

# Set up the figure
fig, axes = plt.subplots(1, 4, figsize=(16, 5))

def draw_network(ax, title, dropped_neurons=None, is_inference=False):
    """Draw a simple 3-layer network with optional dropped neurons."""
    ax.set_xlim(-1, 4)
    ax.set_ylim(-0.5, 4.5)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Layer positions
    layers = [
        [(0, 3.5), (0, 2.5), (0, 1.5), (0, 0.5)],  # Input (4 neurons)
        [(1.5, 3), (1.5, 2), (1.5, 1)],  # Hidden (3 neurons)
        [(3, 2.5), (3, 1.5)],  # Output (2 neurons)
    ]
    
    dropped = dropped_neurons or []
    
    # Draw connections first (so they're behind neurons)
    for l, layer in enumerate(layers[:-1]):
        next_layer = layers[l + 1]
        for i, (x1, y1) in enumerate(layer):
            for j, (x2, y2) in enumerate(next_layer):
                # Check if either neuron is dropped
                neuron1_dropped = (l, i) in dropped
                neuron2_dropped = (l + 1, j) in dropped
                
                if neuron1_dropped or neuron2_dropped:
                    continue  # Don't draw connection
                
                ax.plot([x1 + 0.2, x2 - 0.2], [y1, y2], 
                       color='#90A4AE', linewidth=1.5, alpha=0.6)
    
    # Draw neurons
    for l, layer in enumerate(layers):
        for i, (x, y) in enumerate(layer):
            is_dropped = (l, i) in dropped
            
            if is_dropped:
                # Draw X for dropped neuron
                color = '#FFCDD2'
                circle = Circle((x, y), 0.2, facecolor=color, edgecolor='#EF5350', 
                               linewidth=2, linestyle='--', alpha=0.5)
                ax.add_patch(circle)
                ax.text(x, y, '×', ha='center', va='center', fontsize=14, 
                       color='#EF5350', fontweight='bold')
            else:
                # Normal neuron
                if l == 0:
                    color = '#BBDEFB'  # Input - blue
                elif l == len(layers) - 1:
                    color = '#C8E6C9'  # Output - green
                else:
                    color = '#FFF9C4' if not is_inference else '#FFF9C4'  # Hidden - yellow
                
                circle = Circle((x, y), 0.2, facecolor=color, edgecolor='#424242', 
                               linewidth=1.5)
                ax.add_patch(circle)
    
    ax.set_title(title, fontsize=11, fontweight='bold', pad=10)

# Training networks with different dropout patterns
ax1 = axes[0]
draw_network(ax1, 'Training: Net 1\n(dropout pattern A)', 
             dropped_neurons=[(1, 1), (0, 2)])

ax2 = axes[1]
draw_network(ax2, 'Training: Net 2\n(dropout pattern B)', 
             dropped_neurons=[(1, 0), (0, 0)])

ax3 = axes[2]
draw_network(ax3, 'Training: Net 3\n(dropout pattern C)', 
             dropped_neurons=[(1, 2), (0, 1), (0, 3)])

# Inference (all neurons active)
ax4 = axes[3]
draw_network(ax4, 'Inference:\nAll neurons active\n(weights scaled by 1-p)', 
             is_inference=True)

# Add explanatory text at bottom
explanation = (
    'With N droppable neurons, dropout implicitly trains 2ᴺ sub-networks!\n'
    'At inference, all neurons are active but weights are scaled by (1-p) to match expected values.\n'
    'This approximates averaging predictions from all sub-networks → ensemble effect.'
)
fig.text(0.5, 0.02, explanation, ha='center', fontsize=10, color='#424242',
         bbox=dict(boxstyle='round', facecolor='#E8F5E9', edgecolor='#81C784', alpha=0.9))

# Legend
legend_elements = [
    mpatches.Patch(facecolor='#BBDEFB', edgecolor='#424242', label='Input layer'),
    mpatches.Patch(facecolor='#FFF9C4', edgecolor='#424242', label='Hidden layer'),
    mpatches.Patch(facecolor='#C8E6C9', edgecolor='#424242', label='Output layer'),
    mpatches.Patch(facecolor='#FFCDD2', edgecolor='#EF5350', label='Dropped neuron'),
]
fig.legend(handles=legend_elements, loc='upper right', fontsize=9, 
           bbox_to_anchor=(0.98, 0.98))

plt.suptitle('Dropout as Implicit Ensemble', fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0.15, 1, 0.92])
plt.savefig('dropout_ensemble.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()

print("Generated dropout_ensemble.png")
