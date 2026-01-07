"""
Generate Word2Vec architecture visualization.
Shows Skip-gram and CBOW architectures for learning word embeddings.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle, FancyArrowPatch
import numpy as np

# Set up the figure
fig, axes = plt.subplots(1, 2, figsize=(14, 7))

def draw_word_box(ax, x, y, word, color='#E3F2FD'):
    """Draw a box representing a word."""
    rect = FancyBboxPatch((x - 0.4, y - 0.2), 0.8, 0.4,
                          boxstyle="round,pad=0.02,rounding_size=0.1",
                          facecolor=color, edgecolor='#424242', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x, y, word, ha='center', va='center', fontsize=10, fontweight='bold')

def draw_layer(ax, x, y, label, color='#FFF9C4'):
    """Draw a layer box."""
    rect = FancyBboxPatch((x - 0.5, y - 0.3), 1.0, 0.6,
                          boxstyle="round,pad=0.02,rounding_size=0.1",
                          facecolor=color, edgecolor='#424242', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x, y, label, ha='center', va='center', fontsize=9)

def draw_arrow(ax, x1, y1, x2, y2, color='#616161'):
    """Draw an arrow."""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.5))

# Left plot: Skip-gram
ax1 = axes[0]
ax1.set_xlim(-1, 6)
ax1.set_ylim(-0.5, 5)
ax1.set_aspect('equal')
ax1.axis('off')
ax1.set_title('Skip-gram: Predict Context from Word', fontsize=12, fontweight='bold', pad=15)

# Input word
draw_word_box(ax1, 0.5, 2.5, '"cat"', color='#BBDEFB')
ax1.text(0.5, 3.3, 'Input\nWord', ha='center', va='center', fontsize=9, color='#424242')

# Embedding layer
draw_layer(ax1, 2.5, 2.5, 'Embedding\n(V × d)', color='#FFF9C4')

# Hidden representation
ax1.add_patch(Circle((4, 2.5), 0.25, facecolor='#C8E6C9', edgecolor='#424242', linewidth=1.5))
ax1.text(4, 2.5, 'h', ha='center', va='center', fontsize=10, fontweight='bold')
ax1.text(4, 3.1, 'Hidden\n(d dims)', ha='center', va='center', fontsize=8, color='#424242')

# Output words (context)
output_words = [(5.5, 4), (5.5, 2.5), (5.5, 1)]
output_labels = ['"the"', '"sat"', '"on"']

for (x, y), label in zip(output_words, output_labels):
    draw_word_box(ax1, x, y, label, color='#FFCDD2')

ax1.text(5.5, 4.8, 'Context Words\n(predict)', ha='center', va='center', fontsize=9, color='#424242')

# Arrows
draw_arrow(ax1, 1, 2.5, 1.9, 2.5)
draw_arrow(ax1, 3.1, 2.5, 3.7, 2.5)
draw_arrow(ax1, 4.3, 2.7, 5, 3.8)
draw_arrow(ax1, 4.3, 2.5, 5, 2.5)
draw_arrow(ax1, 4.3, 2.3, 5, 1.2)

# Right plot: CBOW
ax2 = axes[1]
ax2.set_xlim(-1, 6)
ax2.set_ylim(-0.5, 5)
ax2.set_aspect('equal')
ax2.axis('off')
ax2.set_title('CBOW: Predict Word from Context', fontsize=12, fontweight='bold', pad=15)

# Input context words
input_words = [(0.5, 4), (0.5, 2.5), (0.5, 1)]
input_labels = ['"the"', '"sat"', '"on"']

for (x, y), label in zip(input_words, input_labels):
    draw_word_box(ax2, x, y, label, color='#BBDEFB')

ax2.text(0.5, 4.8, 'Context Words\n(input)', ha='center', va='center', fontsize=9, color='#424242')

# Embedding + Average
draw_layer(ax2, 2.5, 2.5, 'Embed &\nAverage', color='#FFF9C4')

# Hidden representation
ax2.add_patch(Circle((4, 2.5), 0.25, facecolor='#C8E6C9', edgecolor='#424242', linewidth=1.5))
ax2.text(4, 2.5, 'h', ha='center', va='center', fontsize=10, fontweight='bold')
ax2.text(4, 3.1, 'Hidden\n(d dims)', ha='center', va='center', fontsize=8, color='#424242')

# Output word
draw_word_box(ax2, 5.5, 2.5, '"cat"', color='#FFCDD2')
ax2.text(5.5, 3.3, 'Target Word\n(predict)', ha='center', va='center', fontsize=9, color='#424242')

# Arrows
draw_arrow(ax2, 1, 3.8, 1.9, 2.7)
draw_arrow(ax2, 1, 2.5, 1.9, 2.5)
draw_arrow(ax2, 1, 1.2, 1.9, 2.3)
draw_arrow(ax2, 3.1, 2.5, 3.7, 2.5)
draw_arrow(ax2, 4.3, 2.5, 5, 2.5)

# Add explanation at bottom
explanation = """
Training Objective (Skip-gram): Maximize P(context | word) = P("the" | "cat") × P("sat" | "cat") × P("on" | "cat")

How it learns: Words appearing in similar contexts get similar embeddings because they must predict similar context words.

"The cat sat..." and "The dog sat..." → "cat" and "dog" embeddings become similar because they predict the same context.
"""

fig.text(0.5, 0.02, explanation, ha='center', fontsize=10, color='#424242',
         bbox=dict(boxstyle='round', facecolor='#FFF9C4', edgecolor='#FFC107', alpha=0.9),
         wrap=True)

plt.tight_layout(rect=[0, 0.18, 1, 0.95])
plt.savefig('word2vec_architecture.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()

print("Generated word2vec_architecture.png")
