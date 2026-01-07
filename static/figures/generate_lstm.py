#!/usr/bin/env python3
"""Generate LSTM architecture diagram - cleaner version."""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
import matplotlib.patches as mpatches
import numpy as np

plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.size'] = 10

fig, ax = plt.subplots(figsize=(14, 9))

# Colors
forget_color = '#ffcdd2'  # Light red
input_color = '#c8e6c9'  # Light green  
output_color = '#bbdefb'  # Light blue
sigmoid_color = '#fff3e0'  # Light orange
tanh_color = '#e1bee7'  # Light purple
arrow_color = '#424242'

# Helper functions
def draw_gate(ax, x, y, label, color, width=0.9, height=0.55):
    """Draw a gate box."""
    box = FancyBboxPatch((x - width/2, y - height/2), width, height,
                         boxstyle="round,pad=0.05", facecolor=color,
                         edgecolor='#424242', linewidth=2)
    ax.add_patch(box)
    ax.text(x, y, label, ha='center', va='center', fontsize=13, fontweight='bold')

def draw_op(ax, x, y, label, color='white', size=0.35):
    """Draw an operation circle."""
    circle = Circle((x, y), size, facecolor=color, edgecolor='#424242', linewidth=2, zorder=10)
    ax.add_patch(circle)
    ax.text(x, y, label, ha='center', va='center', fontsize=14, fontweight='bold', zorder=11)

def arrow(ax, start, end, color=arrow_color, lw=2, zorder=5):
    """Draw an arrow."""
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle='->', color=color, lw=lw),
                zorder=zorder)

# Layout - wider spacing
cell_y = 7.5
gate_y = 4.0
mult_y = 5.5  # intermediate multiplication level
hidden_y = 1.5

# === CELL STATE HIGHWAY (TOP) ===
ax.annotate('', xy=(14, cell_y), xytext=(0.5, cell_y),
            arrowprops=dict(arrowstyle='->', color='#43a047', lw=5))
ax.text(-0.3, cell_y, r'$c_{t-1}$', fontsize=15, va='center', fontweight='bold')
ax.text(14.5, cell_y, r'$c_t$', fontsize=15, va='center', fontweight='bold')
ax.text(7, cell_y + 0.8, 'Cell State (Long-term memory)', fontsize=12, ha='center', 
        style='italic', color='#43a047', fontweight='bold')

# === FORGET GATE (LEFT) ===
forget_x = 3
draw_gate(ax, forget_x, gate_y, r'$\sigma$', sigmoid_color)
ax.text(forget_x, gate_y - 1.0, 'Forget\nGate', ha='center', fontsize=10, 
        color='#c62828', fontweight='bold')

# Forget gate multiply on cell state
draw_op(ax, forget_x, cell_y, '×', forget_color)

# Arrow from forget gate to multiply
arrow(ax, (forget_x, gate_y + 0.35), (forget_x, cell_y - 0.4))

# === INPUT GATE + CANDIDATE (MIDDLE) ===
input_x = 6
tanh_x = 8

# Input gate sigmoid
draw_gate(ax, input_x, gate_y, r'$\sigma$', sigmoid_color)
ax.text(input_x, gate_y - 1.0, 'Input\nGate', ha='center', fontsize=10, 
        color='#2e7d32', fontweight='bold')

# Candidate tanh
draw_gate(ax, tanh_x, gate_y, 'tanh', tanh_color)
ax.text(tanh_x, gate_y - 1.0, 'Candidate\n' + r'$\tilde{c}_t$', ha='center', fontsize=10, 
        color='#6a1b9a', fontweight='bold')

# Multiply input gate output with candidate
mult_input_x = 7
draw_op(ax, mult_input_x, mult_y, '×', input_color)

# Add to cell state
add_x = 7
draw_op(ax, add_x, cell_y, '+', '#c8e6c9')

# Arrows for input path
arrow(ax, (input_x, gate_y + 0.35), (input_x, mult_y))  # σ up
arrow(ax, (input_x, mult_y), (mult_input_x - 0.4, mult_y))  # σ to multiply
arrow(ax, (tanh_x, gate_y + 0.35), (tanh_x, mult_y))  # tanh up
arrow(ax, (tanh_x, mult_y), (mult_input_x + 0.4, mult_y))  # tanh to multiply
arrow(ax, (mult_input_x, mult_y + 0.4), (add_x, cell_y - 0.4))  # multiply to add

# === OUTPUT GATE (RIGHT) ===
output_x = 11
draw_gate(ax, output_x, gate_y, r'$\sigma$', sigmoid_color)
ax.text(output_x, gate_y - 1.0, 'Output\nGate', ha='center', fontsize=10, 
        color='#1565c0', fontweight='bold')

# Tanh on cell state
tanh_out_x = 11
tanh_out_y = 6.2
draw_gate(ax, tanh_out_x, tanh_out_y, 'tanh', tanh_color, width=0.8, height=0.45)

# Output multiply
output_mult_y = 5.2
draw_op(ax, output_x, output_mult_y, '×', output_color)

# Arrow from output gate to multiply
arrow(ax, (output_x, gate_y + 0.35), (output_x, output_mult_y - 0.4))

# Arrow from cell state down to tanh
arrow(ax, (tanh_out_x, cell_y - 0.1), (tanh_out_x, tanh_out_y + 0.3))
arrow(ax, (tanh_out_x, tanh_out_y - 0.3), (output_x, output_mult_y + 0.4))

# Arrow from multiply down to hidden state output
arrow(ax, (output_x, output_mult_y - 0.4), (output_x, hidden_y + 0.4))

# === HIDDEN STATE FLOW ===
# Draw hidden state line
ax.plot([0.5, 2], [hidden_y, hidden_y], color='#1976d2', lw=4)
ax.text(-0.3, hidden_y, r'$h_{t-1}$', fontsize=15, va='center', fontweight='bold')

# Concatenation point
concat_x = 2.5
concat_y = hidden_y
ax.plot([2, concat_x], [hidden_y, hidden_y], color='#1976d2', lw=4)

# Draw vertical line going up from concat
ax.plot([concat_x, concat_x], [hidden_y, gate_y - 0.35], color='#1976d2', lw=3)

# Horizontal branches to each gate
for gx in [forget_x, input_x, tanh_x, output_x]:
    branch_y = gate_y - 0.35
    ax.plot([concat_x, gx], [branch_y, branch_y], color='#1976d2', lw=2, ls='-', alpha=0.7)
    arrow(ax, (gx, branch_y), (gx, gate_y - 0.35))

# x_t input
input_arrow_x = 5
ax.text(input_arrow_x, 0, r'$x_t$', fontsize=15, ha='center', fontweight='bold')
arrow(ax, (input_arrow_x, 0.3), (input_arrow_x, hidden_y - 0.3))
ax.plot([input_arrow_x, concat_x], [hidden_y - 0.3, hidden_y - 0.3], color='#666', lw=2, ls='--')

# Output hidden state
ax.plot([output_x, 14], [hidden_y, hidden_y], color='#1976d2', lw=4)
ax.text(14.5, hidden_y, r'$h_t$', fontsize=15, va='center', fontweight='bold')

# Copy h_t to output 
ax.annotate('', xy=(14, hidden_y), xytext=(output_x + 0.5, hidden_y),
            arrowprops=dict(arrowstyle='->', color='#1976d2', lw=4))

# === LEGEND ===
legend_y = -1.0
legend_items = [
    (2, legend_y, '×', 'Element-wise multiply'),
    (5.5, legend_y, '+', 'Element-wise add'),
    (9, legend_y, 'σ', 'Sigmoid (0-1)'),
    (12.5, legend_y, 'tanh', 'Tanh (-1 to 1)'),
]

for x, y, sym, label in legend_items:
    if sym in ['×', '+']:
        draw_op(ax, x, y, sym, 'white', size=0.28)
    else:
        draw_gate(ax, x, y, sym, sigmoid_color if sym == 'σ' else tanh_color, width=0.7, height=0.4)
    ax.text(x + 0.7, y, label, fontsize=10, va='center')

# Title
ax.set_title('LSTM Cell Architecture', fontsize=18, fontweight='bold', pad=15)

ax.set_xlim(-1, 16)
ax.set_ylim(-2, 9)
ax.set_aspect('equal')
ax.axis('off')

plt.tight_layout()
plt.savefig('lstm_architecture.png', bbox_inches='tight', facecolor='white', dpi=150)
plt.close()

print("Generated lstm_architecture.png")
