#!/usr/bin/env python3
"""Generate MLP Architecture Diagram"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

def draw_mlp():
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis('off')
    
    # Layer positions
    input_x = 1.5
    hidden_x = 5
    output_x = 8.5
    
    # Node positions
    input_nodes = [(input_x, 5.5), (input_x, 3.5), (input_x, 1.5)]
    hidden_nodes = [(hidden_x, 5), (hidden_x, 3.5), (hidden_x, 2)]
    output_nodes = [(output_x, 3.5)]
    
    node_radius = 0.35
    
    # Colors
    input_color = '#4ECDC4'
    hidden_color = '#FF6B6B'
    output_color = '#45B7D1'
    
    # Draw connections (weights)
    for ix, iy in input_nodes:
        for hx, hy in hidden_nodes:
            ax.annotate('', xy=(hx - node_radius, hy), xytext=(ix + node_radius, iy),
                       arrowprops=dict(arrowstyle='-', color='#888888', lw=1, alpha=0.6))
    
    for hx, hy in hidden_nodes:
        for ox, oy in output_nodes:
            ax.annotate('', xy=(ox - node_radius, oy), xytext=(hx + node_radius, hy),
                       arrowprops=dict(arrowstyle='-', color='#888888', lw=1, alpha=0.6))
    
    # Draw nodes
    for i, (x, y) in enumerate(input_nodes):
        circle = plt.Circle((x, y), node_radius, color=input_color, ec='#333333', lw=2, zorder=3)
        ax.add_patch(circle)
        ax.text(x, y, f'$x_{i+1}$', ha='center', va='center', fontsize=14, fontweight='bold', zorder=4)
    
    for i, (x, y) in enumerate(hidden_nodes):
        circle = plt.Circle((x, y), node_radius, color=hidden_color, ec='#333333', lw=2, zorder=3)
        ax.add_patch(circle)
        ax.text(x, y, f'$h_{i+1}$', ha='center', va='center', fontsize=14, fontweight='bold', zorder=4)
    
    for i, (x, y) in enumerate(output_nodes):
        circle = plt.Circle((x, y), node_radius, color=output_color, ec='#333333', lw=2, zorder=3)
        ax.add_patch(circle)
        ax.text(x, y, r'$\hat{y}$', ha='center', va='center', fontsize=14, fontweight='bold', zorder=4)
    
    # Layer labels
    ax.text(input_x, 6.5, 'Input Layer', ha='center', fontsize=12, fontweight='bold', color='#333333')
    ax.text(hidden_x, 6.5, 'Hidden Layer', ha='center', fontsize=12, fontweight='bold', color='#333333')
    ax.text(output_x, 6.5, 'Output Layer', ha='center', fontsize=12, fontweight='bold', color='#333333')
    
    # Weight labels (positioned below arrows)
    ax.text(3.25, 0.3, r'$W_1, b_1$', ha='center', fontsize=12, fontstyle='italic', color='#555555')
    ax.text(6.75, 0.3, r'$W_2, b_2$', ha='center', fontsize=12, fontstyle='italic', color='#555555')
    
    # Arrows indicating weight regions (positioned above labels)
    ax.annotate('', xy=(4, 0.7), xytext=(2.5, 0.7),
               arrowprops=dict(arrowstyle='<->', color='#555555', lw=1.5))
    ax.annotate('', xy=(8, 0.7), xytext=(5.5, 0.7),
               arrowprops=dict(arrowstyle='<->', color='#555555', lw=1.5))
    
    plt.tight_layout()
    plt.savefig('mlp_architecture.png', dpi=150, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print("Saved mlp_architecture.png")
    plt.close()

if __name__ == "__main__":
    draw_mlp()
