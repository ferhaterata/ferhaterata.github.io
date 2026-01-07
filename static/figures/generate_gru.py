#!/usr/bin/env python3
"""Generate GRU architecture diagram."""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle
import numpy as np

plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.size'] = 11

fig, ax = plt.subplots(figsize=(12, 7))

# Colors
reset_color = '#ffcdd2'  # Light red
update_color = '#c8e6c9'  # Light green  
sigmoid_color = '#fff3e0'  # Light orange
tanh_color = '#e1bee7'  # Light purple
arrow_color = '#424242'

# Helper functions
def draw_gate(ax, x, y, label, color, width=0.8, height=0.5):
    """Draw a gate box."""
    box = FancyBboxPatch((x - width/2, y - height/2), width, height,
                         boxstyle="round,pad=0.05", facecolor=color,
                         edgecolor='#424242', linewidth=2)
    ax.add_patch(box)
    ax.text(x, y, label, ha='center', va='center', fontsize=12, fontweight='bold')

def draw_op(ax, x, y, label, color='white', size=0.3):
    """Draw an operation circle."""
    circle = Circle((x, y), size, facecolor=color, edgecolor='#424242', linewidth=2)
    ax.add_patch(circle)
    ax.text(x, y, label, ha='center', va='center', fontsize=14, fontweight='bold')

def arrow(ax, start, end, color=arrow_color, style='->', lw=2):
    """Draw an arrow."""
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle=style, color=color, lw=lw))

# Layout constants
left_x = 1
right_x = 11
hidden_y = 2.5
gate_y = 4.5
input_y = 0.5

# Draw hidden state input/output (main highway)
ax.annotate('', xy=(right_x + 0.5, hidden_y), xytext=(left_x - 0.5, hidden_y),
            arrowprops=dict(arrowstyle='->', color='#1976d2', lw=4))
ax.text(left_x - 1.2, hidden_y, r'$h_{t-1}$', fontsize=14, va='center', fontweight='bold')
ax.text(right_x + 1, hidden_y, r'$h_t$', fontsize=14, va='center', fontweight='bold')

# Input xt
ax.text(6, input_y - 0.3, r'$x_t$', fontsize=14, ha='center', fontweight='bold')
arrow(ax, (6, input_y), (6, hidden_y - 0.5))

# === RESET GATE ===
reset_x = 3
draw_gate(ax, reset_x, gate_y, r'$\sigma$', sigmoid_color)
ax.text(reset_x, gate_y + 1.0, 'Reset\nGate', ha='center', fontsize=10, color='#c62828', fontweight='bold')

# Reset gate multiply
draw_op(ax, reset_x, 3.3, '×', reset_color, size=0.35)

# Arrows for reset gate
arrow(ax, (reset_x, hidden_y), (reset_x, 3.3 - 0.4))
arrow(ax, (reset_x, gate_y - 0.3), (reset_x, 3.3 + 0.4))

# === UPDATE GATE ===
update_x = 5.5
draw_gate(ax, update_x, gate_y, r'$\sigma$', sigmoid_color)
ax.text(update_x, gate_y + 1.0, 'Update\nGate', ha='center', fontsize=10, color='#2e7d32', fontweight='bold')

# === CANDIDATE HIDDEN STATE ===
tanh_x = 7
draw_gate(ax, tanh_x, gate_y, 'tanh', tanh_color)
ax.text(tanh_x, gate_y + 1.0, 'Candidate\n' + r'$\tilde{h}_t$', ha='center', fontsize=10, color='#6a1b9a', fontweight='bold')

# === FINAL COMPUTATION ===
# (1 - z) * h_{t-1}
one_minus_z_x = 8.5
draw_op(ax, one_minus_z_x, 3.6, '1-', update_color, size=0.35)
ax.text(one_minus_z_x + 0.1, 3.6 + 0.6, r'$1 - z_t$', fontsize=10, ha='center')

# Multiply (1-z) * h_{t-1}
mult1_x = 8.5
draw_op(ax, mult1_x, hidden_y, '×', update_color, size=0.35)

# z * candidate
mult2_x = 9.5
draw_op(ax, mult2_x, 4.0, '×', '#e1bee7', size=0.35)

# Final add
add_x = 9.5
draw_op(ax, add_x, hidden_y, '+', '#bbdefb', size=0.35)

# Arrows for update gate path
arrow(ax, (update_x, gate_y - 0.3), (update_x, 3.6))
arrow(ax, (update_x, 3.6), (one_minus_z_x - 0.4, 3.6))
arrow(ax, (one_minus_z_x, 3.25), (mult1_x, hidden_y + 0.4))

# Arrow from update gate to z * candidate
arrow(ax, (update_x + 0.4, 3.6), (mult2_x - 0.4, 4.0))

# Arrow from candidate to multiply
arrow(ax, (tanh_x, gate_y - 0.3), (tanh_x, 4.0))
arrow(ax, (tanh_x, 4.0), (mult2_x - 0.4, 4.0))
arrow(ax, (mult2_x, 3.65), (add_x, hidden_y + 0.4))

# Connect from reset gate output to tanh input (via h_{t-1})
arrow(ax, (reset_x + 0.4, 3.3), (tanh_x - 0.4, 3.3))
ax.plot([tanh_x - 0.4, tanh_x - 0.4], [3.3, gate_y - 0.3], color=arrow_color, lw=2)

# Connect hidden line
ax.plot([left_x - 0.5, 2.5], [hidden_y, hidden_y], color='#1976d2', lw=4)

# Fan out from hidden to gates
ax.plot([2.5, 2.5], [hidden_y, hidden_y - 0.3], color='#1976d2', lw=2)
for gx in [reset_x, update_x]:
    ax.plot([2.5, gx], [hidden_y - 0.3, hidden_y - 0.3], color='#1976d2', lw=2, ls='--', alpha=0.5)

# Output path continues
ax.plot([add_x, right_x + 0.5], [hidden_y, hidden_y], color='#1976d2', lw=4)

# Legend
legend_y = -0.3
legend_items = [
    (2, legend_y, '×', 'Element-wise multiply'),
    (5, legend_y, '+', 'Element-wise add'),
    (8, legend_y, 'σ', 'Sigmoid'),
    (10.5, legend_y, 'tanh', 'Tanh'),
]

for x, y, sym, label in legend_items:
    if sym in ['×', '+']:
        draw_op(ax, x, y, sym, 'white', size=0.25)
    else:
        draw_gate(ax, x, y, sym, sigmoid_color if sym == 'σ' else tanh_color, width=0.6, height=0.35)
    ax.text(x + 0.6, y, label, fontsize=9, va='center')

# Title
ax.set_title('GRU Cell Architecture', fontsize=16, fontweight='bold', pad=10)

# Equations
eq_x = 0.5
eq_y = 6.5
ax.text(eq_x, eq_y, r'$z_t = \sigma(W_z \cdot [h_{t-1}, x_t])$', fontsize=10)
ax.text(eq_x, eq_y - 0.5, r'$r_t = \sigma(W_r \cdot [h_{t-1}, x_t])$', fontsize=10)
ax.text(eq_x, eq_y - 1.0, r'$\tilde{h}_t = \tanh(W \cdot [r_t \odot h_{t-1}, x_t])$', fontsize=10)
ax.text(eq_x, eq_y - 1.5, r'$h_t = (1-z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$', fontsize=10)

ax.set_xlim(-0.5, 13)
ax.set_ylim(-1.2, 7)
ax.set_aspect('equal')
ax.axis('off')

plt.tight_layout()
plt.savefig('gru_architecture.png', bbox_inches='tight', facecolor='white', dpi=150)
plt.close()

print("Generated gru_architecture.png")
