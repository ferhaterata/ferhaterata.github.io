"""
Generate visualization comparing Batch Normalization vs Layer Normalization.
Shows which dimensions each method normalizes over using a tensor grid visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13

# Colors
color_batch = '#4CAF50'      # Green for batch norm
color_layer = '#2196F3'      # Blue for layer norm
color_neutral = '#E0E0E0'    # Gray for unhighlighted
color_highlight = '#FFD54F'  # Yellow highlight
color_text = '#333333'

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Tensor dimensions
n_batch = 4
n_features = 6

def draw_tensor_grid(ax, title, highlight_func, norm_direction, arrow_color):
    """Draw a 2D tensor grid with highlighting based on normalization type."""
    ax.set_xlim(-0.5, n_features + 1.5)
    ax.set_ylim(-1.5, n_batch + 0.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    cell_size = 0.9
    
    # Draw grid cells
    for b in range(n_batch):
        for f in range(n_features):
            color = highlight_func(b, f)
            rect = FancyBboxPatch(
                (f + 0.05, n_batch - 1 - b + 0.05), 
                cell_size, cell_size,
                boxstyle="round,pad=0.02,rounding_size=0.1",
                facecolor=color, 
                edgecolor='white',
                linewidth=2
            )
            ax.add_patch(rect)
            # Add value text
            ax.text(f + 0.5, n_batch - 1 - b + 0.5, f'x{b}{f}', 
                   ha='center', va='center', fontsize=8, color='#555555')
    
    # Axis labels
    ax.text(n_features/2, n_batch + 0.3, 'Features (Hidden Dim)', 
            ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.text(-0.7, n_batch/2 - 0.5, 'Batch', ha='right', va='center', 
            fontsize=11, fontweight='bold', rotation=90)
    
    # Feature indices
    for f in range(n_features):
        ax.text(f + 0.5, -0.3, f'f{f}', ha='center', va='top', fontsize=9, color='#666666')
    
    # Batch indices
    for b in range(n_batch):
        ax.text(-0.2, n_batch - 1 - b + 0.5, f'b{b}', ha='right', va='center', fontsize=9, color='#666666')
    
    # Add normalization direction arrows
    if norm_direction == 'batch':
        # Vertical arrow (across batches)
        ax.annotate('', xy=(n_features + 0.7, 0), xytext=(n_features + 0.7, n_batch - 0.5),
                   arrowprops=dict(arrowstyle='<->', color=arrow_color, lw=3))
        ax.text(n_features + 1.0, n_batch/2 - 0.25, 'Normalize\nacross\nbatch', 
               ha='left', va='center', fontsize=10, color=arrow_color, fontweight='bold')
    else:  # layer
        # Horizontal arrow (across features)
        ax.annotate('', xy=(n_features - 0.5, -0.8), xytext=(0.5, -0.8),
                   arrowprops=dict(arrowstyle='<->', color=arrow_color, lw=3))
        ax.text(n_features/2, -1.2, 'Normalize across features', 
               ha='center', va='top', fontsize=10, color=arrow_color, fontweight='bold')

# === Left Panel: Batch Normalization ===
def batch_norm_highlight(b, f):
    """Highlight one feature column (all batches for feature f=2)."""
    highlight_feature = 2
    if f == highlight_feature:
        return color_batch
    return color_neutral

draw_tensor_grid(axes[0], 'Batch Normalization', batch_norm_highlight, 'batch', color_batch)

# Add explanation text for BatchNorm
batchnorm_text = (
    "• Compute μ, σ² across batch dimension\n"
    "• Same statistics for all samples\n"
    "• Per-feature normalization\n"
    "• Depends on batch size\n"
    "• Issues with small batches"
)
axes[0].text(0.5, -0.25, batchnorm_text, transform=axes[0].transAxes, 
             fontsize=10, va='top', ha='center',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                      edgecolor=color_batch, alpha=0.9))

# Formula (positioned lower to avoid overlap)
axes[0].text(0.5, -0.15, r'$\hat{x}_{b,f} = \frac{x_{b,f} - \mu_f}{\sqrt{\sigma_f^2 + \epsilon}}$', 
             transform=axes[0].transAxes, fontsize=12, va='bottom', ha='center',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#E8F5E9', edgecolor=color_batch))


# === Right Panel: Layer Normalization ===
def layer_norm_highlight(b, f):
    """Highlight one batch row (all features for batch b=1)."""
    highlight_batch = 1
    if b == highlight_batch:
        return color_layer
    return color_neutral

draw_tensor_grid(axes[1], 'Layer Normalization', layer_norm_highlight, 'layer', color_layer)

# Add explanation text for LayerNorm
layernorm_text = (
    "• Compute μ, σ² across feature dimension\n"
    "• Different statistics per sample\n"
    "• Per-sample normalization\n"
    "• Independent of batch size\n"
    "• Preferred for Transformers/NLP"
)
axes[1].text(0.5, -0.25, layernorm_text, transform=axes[1].transAxes, 
             fontsize=10, va='top', ha='center',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                      edgecolor=color_layer, alpha=0.9))

# Formula (positioned lower to avoid overlap with arrows)
axes[1].text(0.5, -0.15, r'$\hat{x}_{b,f} = \frac{x_{b,f} - \mu_b}{\sqrt{\sigma_b^2 + \epsilon}}$', 
             transform=axes[1].transAxes, fontsize=12, va='bottom', ha='center',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#E3F2FD', edgecolor=color_layer))


# Main title with space below
fig.suptitle('Batch Normalization vs Layer Normalization', 
             fontsize=16, fontweight='bold', y=1.02)

# Add subtitle with clear space from title and from content
fig.text(0.5, 0.96, 'Highlighted cells show which elements are normalized together', 
         ha='center', fontsize=11, style='italic', color='#666666')

plt.tight_layout()
plt.subplots_adjust(top=0.86, bottom=0.22, wspace=0.05)  # Space for title area, minimal horizontal gap
plt.savefig('batch_layer_norm.png', dpi=150, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.close()

print("Generated: batch_layer_norm.png")
