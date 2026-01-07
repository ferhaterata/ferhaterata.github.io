#!/usr/bin/env python3
"""
Figure Generator for ML Interview Comprehensive Guide

This script generates all the PNG figures referenced in the companion markdown file:
    ml-prep/ml_interview_comprehensive_guide.md

PURPOSE:
--------
The ML Interview Comprehensive Guide uses these programmatically generated figures
to illustrate key machine learning concepts. By generating figures with code (rather
than static images), we ensure:
1. Consistency in style across all visualizations
2. Easy modification of colors, labels, and data
3. Reproducibility - anyone can regenerate identical figures
4. Version control friendly - the source code IS the specification

FIGURES GENERATED:
------------------
1. sigmoid_function.png      - Sigmoid activation with derivative inset
2. activation_functions.png  - Comparison of 6 activation functions (sigmoid, tanh, ReLU, etc.)
3. loss_landscape.png        - 3D loss surface + gradient descent path visualization
4. learning_curves.png       - Underfitting vs good fit vs overfitting diagnosis
5. bias_variance.png         - Polynomial fitting showing bias-variance tradeoff
6. softmax_temperature.png   - Effect of temperature on softmax distributions
7. attention_heatmap.png     - Self-attention weights visualization
8. lr_schedules.png          - Learning rate schedule comparison (cosine, step, warmup)
9. gradient_flow.png         - Vanishing/exploding gradient visualization
10. cross_entropy_vs_mse.png - Why cross-entropy is better for classification

USAGE:
------
    cd ml-prep/figures
    python generate_figures.py

    Or from the ml-prep directory:
    python figures/generate_figures.py

REQUIREMENTS:
-------------
    pip install matplotlib numpy

STYLE:
------
All figures use a consistent style defined by:
- seaborn-v0_8-whitegrid theme
- 150 DPI resolution
- Consistent color palette (COLORS dict)
- White background for markdown embedding

TO MODIFY:
----------
1. Edit the corresponding generate_*() function
2. Run the script to regenerate all figures
3. Changes will automatically appear in the markdown when viewed

Author: Generated for ML Interview Prep
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# Set style for all figures
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.facecolor'] = 'white'

# Color palette
COLORS = {
    'primary': '#2563eb',      # Blue
    'secondary': '#dc2626',    # Red
    'tertiary': '#16a34a',     # Green
    'quaternary': '#9333ea',   # Purple
    'orange': '#ea580c',       # Orange
    'gray': '#6b7280',         # Gray
}


def sigmoid(x):
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-x))


def tanh_func(x):
    """Tanh activation function."""
    return np.tanh(x)


def relu(x):
    """ReLU activation function."""
    return np.maximum(0, x)


def leaky_relu(x, alpha=0.1):
    """Leaky ReLU activation function."""
    return np.where(x > 0, x, alpha * x)


def gelu(x):
    """GELU activation function (approximation)."""
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def swish(x):
    """Swish/SiLU activation function."""
    return x * sigmoid(x)


def softmax(x, temperature=1.0):
    """Softmax with temperature."""
    x = x / temperature
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()


def generate_sigmoid_figure():
    """Generate sigmoid function visualization."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.linspace(-8, 8, 500)
    y = sigmoid(x)
    
    # Main sigmoid curve
    ax.plot(x, y, color=COLORS['primary'], linewidth=2.5, label='σ(z) = 1/(1+e⁻ᶻ)')
    
    # Key points
    key_points = [
        (0, 0.5, 'z=0: σ(0)=0.5\n(max uncertainty)'),
        (4, sigmoid(4), f'z=4: σ(4)≈{sigmoid(4):.3f}'),
        (-4, sigmoid(-4), f'z=-4: σ(-4)≈{sigmoid(-4):.3f}'),
    ]
    
    for xp, yp, label in key_points:
        ax.scatter([xp], [yp], color=COLORS['secondary'], s=80, zorder=5)
        if xp == 0:
            ax.annotate(label, (xp, yp), xytext=(1.5, 0.5), fontsize=10,
                       arrowprops=dict(arrowstyle='->', color='gray', lw=1))
        elif xp > 0:
            ax.annotate(label, (xp, yp), xytext=(xp+0.5, yp-0.15), fontsize=9)
        else:
            ax.annotate(label, (xp, yp), xytext=(xp-2.5, yp+0.1), fontsize=9)
    
    # Asymptotes
    ax.axhline(y=1, color=COLORS['gray'], linestyle='--', alpha=0.5, label='y → 1 as z → +∞')
    ax.axhline(y=0, color=COLORS['gray'], linestyle='--', alpha=0.5, label='y → 0 as z → -∞')
    ax.axhline(y=0.5, color=COLORS['orange'], linestyle=':', alpha=0.7, label='y = 0.5 (decision boundary)')
    ax.axvline(x=0, color=COLORS['gray'], linestyle=':', alpha=0.3)
    
    # Derivative visualization (small subplot region)
    ax_inset = fig.add_axes([0.15, 0.55, 0.25, 0.3])
    y_deriv = y * (1 - y)  # σ'(z) = σ(z)(1-σ(z))
    ax_inset.plot(x, y_deriv, color=COLORS['tertiary'], linewidth=2)
    ax_inset.set_title("Derivative σ'(z)", fontsize=10)
    ax_inset.set_xlabel('z', fontsize=9)
    ax_inset.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
    ax_inset.set_xlim(-6, 6)
    ax_inset.text(0.5, 0.22, 'max at z=0', fontsize=8, ha='left')
    
    ax.set_xlabel('z = w·x + b (pre-activation)', fontsize=12)
    ax.set_ylabel('σ(z) = P(y=1|x)', fontsize=12)
    ax.set_title('Sigmoid Activation Function', fontsize=14, fontweight='bold')
    ax.set_xlim(-8, 8)
    ax.set_ylim(-0.1, 1.1)
    ax.legend(loc='lower right', framealpha=0.9)
    
    # Add properties box
    props_text = (
        "Properties:\n"
        "• Range: (0, 1)\n"
        "• σ(-z) = 1 - σ(z)\n"
        "• σ'(z) = σ(z)(1-σ(z))\n"
        "• Max gradient at z=0"
    )
    ax.text(0.02, 0.98, props_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('sigmoid_function.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Generated sigmoid_function.png")


def generate_activation_functions():
    """Generate comparison of activation functions."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    x = np.linspace(-5, 5, 500)
    
    activations = [
        ('Sigmoid', sigmoid(x), "σ(x) = 1/(1+e⁻ˣ)", COLORS['primary'], 'Range: (0, 1)\nVanishing gradient for |x| >> 0'),
        ('Tanh', tanh_func(x), "tanh(x)", COLORS['secondary'], 'Range: (-1, 1)\nZero-centered output'),
        ('ReLU', relu(x), "max(0, x)", COLORS['tertiary'], 'Range: [0, ∞)\nDead neurons if x < 0'),
        ('Leaky ReLU', leaky_relu(x), "max(0.1x, x)", COLORS['quaternary'], 'No dead neurons\nSmall negative gradient'),
        ('GELU', gelu(x), "x·Φ(x)", COLORS['orange'], 'Smooth approximation\nUsed in Transformers'),
        ('Swish/SiLU', swish(x), "x·σ(x)", '#0891b2', 'Self-gated\nSmooth, non-monotonic'),
    ]
    
    for ax, (name, y, formula, color, notes) in zip(axes.flat, activations):
        ax.plot(x, y, color=color, linewidth=2.5, label=formula)
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3, linewidth=0.5)
        ax.axvline(x=0, color='gray', linestyle='-', alpha=0.3, linewidth=0.5)
        ax.set_title(name, fontsize=13, fontweight='bold')
        ax.set_xlabel('x')
        ax.set_ylabel('f(x)')
        ax.legend(loc='upper left', fontsize=9)
        ax.set_xlim(-5, 5)
        
        # Add notes
        ax.text(0.98, 0.02, notes, transform=ax.transAxes, fontsize=8,
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.suptitle('Activation Functions Comparison', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('activation_functions.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Generated activation_functions.png")


def generate_loss_landscape():
    """Generate 3D loss landscape visualization."""
    fig = plt.figure(figsize=(14, 5))
    
    # 3D surface plot
    ax1 = fig.add_subplot(121, projection='3d')
    
    w1 = np.linspace(-3, 3, 100)
    w2 = np.linspace(-3, 3, 100)
    W1, W2 = np.meshgrid(w1, w2)
    
    # Create a loss landscape with multiple minima
    L = 0.5 * (W1**2 + W2**2) + 0.3 * np.sin(3*W1) * np.cos(3*W2)
    
    surf = ax1.plot_surface(W1, W2, L, cmap='viridis', alpha=0.8, edgecolor='none')
    ax1.set_xlabel('w₁')
    ax1.set_ylabel('w₂')
    ax1.set_zlabel('Loss')
    ax1.set_title('Loss Landscape (3D)', fontsize=12, fontweight='bold')
    ax1.view_init(elev=30, azim=45)
    
    # Contour plot with gradient descent path
    ax2 = fig.add_subplot(122)
    
    contour = ax2.contour(W1, W2, L, levels=20, cmap='viridis')
    ax2.clabel(contour, inline=True, fontsize=8)
    
    # Simulate gradient descent path
    path_w1 = [2.5]
    path_w2 = [2.5]
    lr = 0.1
    
    for _ in range(30):
        w1_curr, w2_curr = path_w1[-1], path_w2[-1]
        # Gradient (approximate)
        grad_w1 = w1_curr + 0.9 * np.cos(3*w1_curr) * np.cos(3*w2_curr)
        grad_w2 = w2_curr - 0.9 * np.sin(3*w1_curr) * np.sin(3*w2_curr)
        
        path_w1.append(w1_curr - lr * grad_w1)
        path_w2.append(w2_curr - lr * grad_w2)
    
    ax2.plot(path_w1, path_w2, 'ro-', markersize=4, linewidth=1.5, label='Gradient Descent Path')
    ax2.scatter([path_w1[0]], [path_w2[0]], color='red', s=100, marker='*', zorder=5, label='Start')
    ax2.scatter([path_w1[-1]], [path_w2[-1]], color='green', s=100, marker='o', zorder=5, label='End')
    
    ax2.set_xlabel('w₁')
    ax2.set_ylabel('w₂')
    ax2.set_title('Gradient Descent Path (Contour View)', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.set_xlim(-3, 3)
    ax2.set_ylim(-3, 3)
    
    plt.tight_layout()
    plt.savefig('loss_landscape.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Generated loss_landscape.png")


def generate_learning_curves():
    """Generate learning curves showing overfitting vs good fit."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    epochs = np.arange(1, 101)
    
    # Underfitting
    ax = axes[0]
    train_loss = 2 - 0.5 * np.log(epochs) + np.random.randn(100) * 0.05
    val_loss = 2.2 - 0.4 * np.log(epochs) + np.random.randn(100) * 0.05
    ax.plot(epochs, train_loss, color=COLORS['primary'], linewidth=2, label='Training Loss')
    ax.plot(epochs, val_loss, color=COLORS['secondary'], linewidth=2, label='Validation Loss')
    ax.fill_between(epochs, train_loss, val_loss, alpha=0.1, color='gray')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.set_title('Underfitting (High Bias)', fontsize=12, fontweight='bold')
    ax.legend()
    ax.set_ylim(0, 3)
    ax.text(0.5, 0.95, 'Both losses remain high\nModel too simple', transform=ax.transAxes,
            fontsize=9, verticalalignment='top', horizontalalignment='center',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    # Good fit
    ax = axes[1]
    train_loss = 1.5 * np.exp(-epochs/20) + 0.1 + np.random.randn(100) * 0.02
    val_loss = 1.6 * np.exp(-epochs/25) + 0.15 + np.random.randn(100) * 0.03
    ax.plot(epochs, train_loss, color=COLORS['primary'], linewidth=2, label='Training Loss')
    ax.plot(epochs, val_loss, color=COLORS['secondary'], linewidth=2, label='Validation Loss')
    ax.fill_between(epochs, train_loss, val_loss, alpha=0.1, color='gray')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.set_title('Good Fit', fontsize=12, fontweight='bold')
    ax.legend()
    ax.set_ylim(0, 2)
    ax.text(0.5, 0.95, 'Both losses converge\nSmall gap maintained', transform=ax.transAxes,
            fontsize=9, verticalalignment='top', horizontalalignment='center',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    # Overfitting
    ax = axes[2]
    train_loss = 1.5 * np.exp(-epochs/15) + 0.02 + np.random.randn(100) * 0.01
    val_loss = np.where(epochs < 30,
                        1.5 * np.exp(-epochs/20) + 0.1,
                        0.3 + 0.01 * (epochs - 30) + np.random.randn(100) * 0.02)
    ax.plot(epochs, train_loss, color=COLORS['primary'], linewidth=2, label='Training Loss')
    ax.plot(epochs, val_loss, color=COLORS['secondary'], linewidth=2, label='Validation Loss')
    ax.fill_between(epochs, train_loss, val_loss, alpha=0.1, color='red')
    ax.axvline(x=30, color='gray', linestyle='--', alpha=0.5, label='Early stopping point')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.set_title('Overfitting (High Variance)', fontsize=12, fontweight='bold')
    ax.legend()
    ax.set_ylim(0, 2)
    ax.text(0.5, 0.95, 'Validation loss increases\nafter initial decrease', transform=ax.transAxes,
            fontsize=9, verticalalignment='top', horizontalalignment='center',
            bbox=dict(boxstyle='round', facecolor='lightsalmon', alpha=0.8))
    
    plt.suptitle('Learning Curves: Diagnosing Model Performance', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('learning_curves.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Generated learning_curves.png")


def generate_bias_variance():
    """Generate bias-variance tradeoff visualization."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # True function
    np.random.seed(42)
    x = np.linspace(0, 1, 100)
    x_data = np.random.uniform(0, 1, 20)
    y_true = np.sin(2 * np.pi * x)
    y_data = np.sin(2 * np.pi * x_data) + np.random.randn(20) * 0.3
    
    # Underfitting (degree 1)
    ax = axes[0]
    coeffs = np.polyfit(x_data, y_data, 1)
    y_fit = np.polyval(coeffs, x)
    ax.scatter(x_data, y_data, color=COLORS['primary'], s=50, alpha=0.7, label='Data points')
    ax.plot(x, y_true, color=COLORS['gray'], linestyle='--', linewidth=2, label='True function')
    ax.plot(x, y_fit, color=COLORS['secondary'], linewidth=2.5, label='Linear fit (d=1)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Underfitting\n(High Bias, Low Variance)', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_ylim(-1.8, 1.8)
    
    # Good fit (degree 3)
    ax = axes[1]
    coeffs = np.polyfit(x_data, y_data, 4)
    y_fit = np.polyval(coeffs, x)
    ax.scatter(x_data, y_data, color=COLORS['primary'], s=50, alpha=0.7, label='Data points')
    ax.plot(x, y_true, color=COLORS['gray'], linestyle='--', linewidth=2, label='True function')
    ax.plot(x, y_fit, color=COLORS['tertiary'], linewidth=2.5, label='Polynomial fit (d=4)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Good Fit\n(Balanced Bias-Variance)', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_ylim(-1.8, 1.8)
    
    # Overfitting (degree 15)
    ax = axes[2]
    coeffs = np.polyfit(x_data, y_data, 15)
    y_fit = np.polyval(coeffs, x)
    y_fit = np.clip(y_fit, -3, 3)  # Clip extreme values
    ax.scatter(x_data, y_data, color=COLORS['primary'], s=50, alpha=0.7, label='Data points')
    ax.plot(x, y_true, color=COLORS['gray'], linestyle='--', linewidth=2, label='True function')
    ax.plot(x, y_fit, color=COLORS['quaternary'], linewidth=2.5, label='Polynomial fit (d=15)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Overfitting\n(Low Bias, High Variance)', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_ylim(-1.8, 1.8)
    
    plt.suptitle('Bias-Variance Tradeoff: Model Complexity vs Generalization', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('bias_variance.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Generated bias_variance.png")


def generate_softmax_temperature():
    """Generate softmax temperature visualization."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    
    # Logits
    logits = np.array([2.0, 1.0, 0.5, 0.1, -0.5])
    classes = ['Class A', 'Class B', 'Class C', 'Class D', 'Class E']
    temperatures = [0.5, 1.0, 2.0]
    titles = ['Low Temperature (T=0.5)\n"More confident"', 
              'Normal Temperature (T=1.0)\n"Standard softmax"',
              'High Temperature (T=2.0)\n"More uniform"']
    colors = [COLORS['secondary'], COLORS['primary'], COLORS['tertiary']]
    
    for ax, temp, title, color in zip(axes, temperatures, titles, colors):
        probs = softmax(logits, temperature=temp)
        bars = ax.bar(classes, probs, color=color, alpha=0.8, edgecolor='black', linewidth=1)
        
        # Add probability labels
        for bar, prob in zip(bars, probs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{prob:.3f}', ha='center', fontsize=9)
        
        ax.set_ylabel('Probability')
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_ylim(0, 1.0)
        ax.tick_params(axis='x', rotation=45)
        
        # Add entropy value
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        ax.text(0.95, 0.95, f'Entropy: {entropy:.3f}', transform=ax.transAxes,
                fontsize=9, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.suptitle('Effect of Temperature on Softmax Distribution\n(Same logits: [2.0, 1.0, 0.5, 0.1, -0.5])',
                 fontsize=14, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig('softmax_temperature.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Generated softmax_temperature.png")


def generate_attention_heatmap():
    """Generate attention weights heatmap visualization."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Sample sentence: "The cat sat on the mat."
    # tokens[0]='The', [1]='cat', [2]='sat', [3]='on', [4]='the', [5]='mat', [6]='.'
    tokens = ['The', 'cat', 'sat', 'on', 'the', 'mat', '.']
    n = len(tokens)
    
    # Create realistic attention pattern
    np.random.seed(42)
    attention = np.random.rand(n, n) * 0.05  # Lower base noise
    
    # Semantic attention patterns (what we want to highlight)
    # For "sat" (index 2): should attend to "cat" (subject) and "mat" (location)
    attention[2, 1] = 0.45  # 'sat' → 'cat' (subject performing action)
    attention[2, 5] = 0.35  # 'sat' → 'mat' (related location) ← KEY ADDITION
    attention[2, 0] = 0.05  # 'sat' → 'The' (low weight - not semantically important)
    
    # Other semantic relationships
    attention[1, 2] = 0.3   # 'cat' → 'sat' (what cat does)
    attention[5, 2] = 0.35  # 'mat' → 'sat' (action involving mat)
    attention[5, 1] = 0.25  # 'mat' → 'cat' (who's on the mat)
    attention[3, 2] = 0.4   # 'on' → 'sat' (preposition links to verb)
    attention[3, 5] = 0.3   # 'on' → 'mat' (preposition links to object)
    attention[6, 5] = 0.5   # '.' → 'mat' (end of sentence refs subject)
    
    # Self-attention (diagonal) - moderate self-attention
    np.fill_diagonal(attention, 0.2)
    
    # Normalize rows to sum to 1
    attention = attention / attention.sum(axis=1, keepdims=True)
    
    im = ax.imshow(attention, cmap='Blues', aspect='auto')
    
    # Labels
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(tokens, fontsize=11)
    ax.set_yticklabels(tokens, fontsize=11)
    ax.set_xlabel('Key Position (attending to)', fontsize=12)
    ax.set_ylabel('Query Position (from)', fontsize=12)
    ax.set_title('Self-Attention Weights Visualization\n"The cat sat on the mat."', 
                 fontsize=14, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Attention Weight', fontsize=11)
    
    # Add text annotations
    for i in range(n):
        for j in range(n):
            text = f'{attention[i, j]:.2f}'
            color = 'white' if attention[i, j] > 0.3 else 'black'
            ax.text(j, i, text, ha='center', va='center', color=color, fontsize=9)
    
    plt.tight_layout()
    plt.savefig('attention_heatmap.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Generated attention_heatmap.png")


def generate_lr_schedules():
    """Generate learning rate schedule comparison."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    total_steps = 10000
    warmup_steps = 1000
    steps = np.arange(total_steps)
    max_lr = 1e-3
    min_lr = 1e-5
    
    # Constant
    lr_constant = np.ones(total_steps) * max_lr
    
    # Step decay
    lr_step = np.where(steps < 3000, max_lr,
                np.where(steps < 6000, max_lr * 0.1, max_lr * 0.01))
    
    # Warmup + Cosine
    lr_warmup = np.minimum(steps / warmup_steps, 1.0) * max_lr
    lr_cosine = min_lr + 0.5 * (max_lr - min_lr) * (1 + np.cos(np.pi * np.clip((steps - warmup_steps) / (total_steps - warmup_steps), 0, 1)))
    lr_warmup_cosine = np.where(steps < warmup_steps, lr_warmup, lr_cosine)
    
    # Linear decay with warmup
    lr_linear = np.where(steps < warmup_steps,
                         steps / warmup_steps * max_lr,
                         max_lr * (1 - (steps - warmup_steps) / (total_steps - warmup_steps)))
    lr_linear = np.maximum(lr_linear, min_lr)
    
    ax.plot(steps, lr_constant, label='Constant', color=COLORS['gray'], linewidth=2, linestyle='--')
    ax.plot(steps, lr_step, label='Step Decay', color=COLORS['secondary'], linewidth=2)
    ax.plot(steps, lr_warmup_cosine, label='Warmup + Cosine (LLMs)', color=COLORS['primary'], linewidth=2.5)
    ax.plot(steps, lr_linear, label='Warmup + Linear Decay', color=COLORS['tertiary'], linewidth=2)
    
    ax.axvline(x=warmup_steps, color='gray', linestyle=':', alpha=0.5)
    ax.text(warmup_steps + 100, max_lr * 0.9, 'Warmup ends', fontsize=9, color='gray')
    
    ax.set_xlabel('Training Steps', fontsize=12)
    ax.set_ylabel('Learning Rate', fontsize=12)
    ax.set_title('Learning Rate Schedules Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.set_yscale('log')
    ax.set_ylim(min_lr * 0.5, max_lr * 2)
    ax.grid(True, alpha=0.3)
    
    # Add annotation
    ax.annotate('Warmup prevents\nunstable early updates',
                xy=(500, max_lr * 0.5), xytext=(1500, max_lr * 0.2),
                arrowprops=dict(arrowstyle='->', color='gray'),
                fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('lr_schedules.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Generated lr_schedules.png")


def generate_gradient_flow():
    """Generate gradient flow visualization for vanishing/exploding gradients."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    layers = np.arange(1, 11)
    
    # Vanishing gradients
    ax = axes[0]
    grad_vanish = 0.9 ** layers
    ax.bar(layers, grad_vanish, color=COLORS['secondary'], alpha=0.8, edgecolor='black')
    ax.set_xlabel('Layer (from output)')
    ax.set_ylabel('Relative Gradient Magnitude')
    ax.set_title('Vanishing Gradients\n(σ or tanh, no skip connections)', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.text(0.5, 0.95, 'Each layer: grad × 0.9\nAfter 10 layers: 0.35', transform=ax.transAxes,
            fontsize=9, verticalalignment='top', horizontalalignment='center',
            bbox=dict(boxstyle='round', facecolor='lightsalmon', alpha=0.8))
    
    # Stable gradients (with residuals)
    ax = axes[1]
    grad_stable = np.ones(10) * (0.9 + 0.1 * np.random.randn(10) * 0.1)
    ax.bar(layers, grad_stable, color=COLORS['tertiary'], alpha=0.8, edgecolor='black')
    ax.set_xlabel('Layer (from output)')
    ax.set_ylabel('Relative Gradient Magnitude')
    ax.set_title('Stable Gradients\n(ResNet, LayerNorm)', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1.5)
    ax.text(0.5, 0.95, 'Residual: grad flows through\nidentity path', transform=ax.transAxes,
            fontsize=9, verticalalignment='top', horizontalalignment='center',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    # Exploding gradients
    ax = axes[2]
    grad_explode = 1.1 ** layers
    ax.bar(layers, grad_explode, color=COLORS['quaternary'], alpha=0.8, edgecolor='black')
    ax.set_xlabel('Layer (from output)')
    ax.set_ylabel('Relative Gradient Magnitude')
    ax.set_title('Exploding Gradients\n(Large weights, vanilla RNN)', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 3)
    ax.text(0.5, 0.95, 'Each layer: grad × 1.1\nAfter 10 layers: 2.6', transform=ax.transAxes,
            fontsize=9, verticalalignment='top', horizontalalignment='center',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.suptitle('Gradient Flow Through Deep Networks', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('gradient_flow.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Generated gradient_flow.png")


def generate_overfitting_spectrum():
    """Generate the overfitting spectrum visualization matching ASCII art layout."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # True function
    np.random.seed(42)
    x = np.linspace(0, 1, 200)
    x_data = np.array([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95])
    y_true = np.sin(2.5 * np.pi * x)
    y_data = np.sin(2.5 * np.pi * x_data) + np.random.randn(10) * 0.15
    
    # Panel 1: Underfitting (straight line)
    ax = axes[0]
    coeffs = np.polyfit(x_data, y_data, 1)
    y_fit = np.polyval(coeffs, x)
    ax.scatter(x_data, y_data, color=COLORS['primary'], s=80, alpha=0.8, zorder=5, label='Data')
    ax.plot(x, y_true, color=COLORS['gray'], linestyle='--', linewidth=2, alpha=0.7, label='True function')
    ax.plot(x, y_fit, color=COLORS['secondary'], linewidth=3, label='Model (line)')
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title('UNDERFITTING\n"Too Simple"', fontsize=14, fontweight='bold', color=COLORS['secondary'])
    ax.legend(loc='upper right', fontsize=9)
    ax.set_ylim(-1.8, 1.8)
    ax.text(0.5, -0.15, 'High Bias\nCannot capture pattern', transform=ax.transAxes,
            fontsize=10, ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='lightsalmon', alpha=0.9))
    
    # Panel 2: Good fit (degree 4 polynomial)
    ax = axes[1]
    coeffs = np.polyfit(x_data, y_data, 5)
    y_fit = np.polyval(coeffs, x)
    ax.scatter(x_data, y_data, color=COLORS['primary'], s=80, alpha=0.8, zorder=5, label='Data')
    ax.plot(x, y_true, color=COLORS['gray'], linestyle='--', linewidth=2, alpha=0.7, label='True function')
    ax.plot(x, y_fit, color=COLORS['tertiary'], linewidth=3, label='Model (matches!)')
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title('GOOD FIT\n"Just Right"', fontsize=14, fontweight='bold', color=COLORS['tertiary'])
    ax.legend(loc='upper right', fontsize=9)
    ax.set_ylim(-1.8, 1.8)
    ax.text(0.5, -0.15, 'Low Bias + Low Variance\nGeneralizes well', transform=ax.transAxes,
            fontsize=10, ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.9))
    
    # Panel 3: Overfitting (very wiggly - high degree polynomial)
    ax = axes[2]
    coeffs = np.polyfit(x_data, y_data, 9)  # degree = n_points - 1 for perfect fit
    y_fit = np.polyval(coeffs, x)
    y_fit = np.clip(y_fit, -2.5, 2.5)  # Clip extreme wiggles
    ax.scatter(x_data, y_data, color=COLORS['primary'], s=80, alpha=0.8, zorder=5, label='Data')
    ax.plot(x, y_true, color=COLORS['gray'], linestyle='--', linewidth=2, alpha=0.7, label='True function')
    ax.plot(x, y_fit, color=COLORS['quaternary'], linewidth=3, label='Model (wiggly!)')
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title('OVERFITTING\n"Too Complex"', fontsize=14, fontweight='bold', color=COLORS['quaternary'])
    ax.legend(loc='upper right', fontsize=9)
    ax.set_ylim(-1.8, 1.8)
    ax.text(0.5, -0.15, 'High Variance\nMemorizes noise', transform=ax.transAxes,
            fontsize=10, ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='plum', alpha=0.9))
    
    plt.suptitle('The Overfitting Spectrum: Model Complexity vs. Fit Quality', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('overfitting_spectrum.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Generated overfitting_spectrum.png")


def generate_learning_curves_diagnostic():
    """Generate learning curves diagnostic tool - 2 panel version matching ASCII art."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    np.random.seed(42)
    epochs = np.arange(1, 101)
    
    # Panel 1: Overfitting (gap = problem!)
    ax = axes[0]
    train_loss = 1.5 * np.exp(-epochs/12) + 0.05 + np.random.randn(100) * 0.01
    # Validation initially decreases then increases
    val_loss_base = np.where(epochs < 25,
                              1.5 * np.exp(-epochs/18) + 0.15,
                              0.35 + 0.012 * (epochs - 25))
    val_loss = val_loss_base + np.random.randn(100) * 0.02
    
    ax.plot(epochs, train_loss, color=COLORS['primary'], linewidth=2.5, label='Training loss')
    ax.plot(epochs, val_loss, color=COLORS['secondary'], linewidth=2.5, label='Validation loss')
    
    # Highlight the gap
    ax.fill_between(epochs[30:], train_loss[30:], val_loss[30:], alpha=0.3, color='red', label='Gap = Problem!')
    ax.axvline(x=25, color='gray', linestyle='--', alpha=0.6)
    ax.annotate('Optimal stopping\npoint', xy=(25, 0.35), xytext=(40, 0.7),
                arrowprops=dict(arrowstyle='->', color='gray'),
                fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax.set_xlabel('Epochs', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('OVERFITTING\n(Gap = Problem!)', fontsize=14, fontweight='bold', color=COLORS['secondary'])
    ax.legend(loc='upper right', fontsize=10)
    ax.set_ylim(0, 1.8)
    ax.set_xlim(0, 100)
    
    # Add arrows showing divergence
    ax.annotate('', xy=(90, val_loss[89]), xytext=(90, train_loss[89]),
                arrowprops=dict(arrowstyle='<->', color='red', lw=2))
    ax.text(93, (val_loss[89] + train_loss[89])/2, 'GAP', fontsize=10, color='red', fontweight='bold', va='center')
    
    # Panel 2: Good fit (both converge)
    ax = axes[1]
    train_loss = 1.5 * np.exp(-epochs/18) + 0.1 + np.random.randn(100) * 0.015
    val_loss = 1.6 * np.exp(-epochs/22) + 0.12 + np.random.randn(100) * 0.02
    
    ax.plot(epochs, train_loss, color=COLORS['primary'], linewidth=2.5, label='Training loss')
    ax.plot(epochs, val_loss, color=COLORS['tertiary'], linewidth=2.5, label='Validation loss')
    
    # Highlight convergence
    ax.fill_between(epochs[50:], train_loss[50:], val_loss[50:], alpha=0.2, color='green')
    
    ax.set_xlabel('Epochs', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('GOOD FIT\n(Both Converge)', fontsize=14, fontweight='bold', color=COLORS['tertiary'])
    ax.legend(loc='upper right', fontsize=10)
    ax.set_ylim(0, 1.8)
    ax.set_xlim(0, 100)
    
    # Add annotation showing convergence
    ax.annotate('Both losses\nplateau together', xy=(80, 0.15), xytext=(60, 0.6),
                arrowprops=dict(arrowstyle='->', color='green'),
                fontsize=10, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    plt.suptitle('Learning Curves: Your Diagnostic Tool', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('learning_curves_diagnostic.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Generated learning_curves_diagnostic.png")


def generate_cross_entropy_vs_mse():
    """Generate comparison of cross-entropy vs MSE loss for classification."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # For true label y=1
    p = np.linspace(0.001, 0.999, 500)  # Predicted probability
    
    # MSE loss: (1 - p)^2
    mse = (1 - p) ** 2
    
    # Cross-entropy loss: -log(p)
    ce = -np.log(p)
    
    ax = axes[0]
    ax.plot(p, mse, color=COLORS['secondary'], linewidth=2.5, label='MSE: (y-ŷ)²')
    ax.plot(p, ce, color=COLORS['primary'], linewidth=2.5, label='CE: -log(ŷ)')
    ax.set_xlabel('Predicted Probability ŷ (true label y=1)', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Loss Functions Comparison\n(True Label = 1)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 5)
    ax.axvline(x=0.5, color='gray', linestyle=':', alpha=0.5)
    ax.text(0.52, 4.5, 'Decision boundary', fontsize=9, color='gray')
    
    # Annotate key difference
    ax.annotate('CE punishes confident\nwrong predictions MORE',
                xy=(0.1, -np.log(0.1)), xytext=(0.3, 3.5),
                arrowprops=dict(arrowstyle='->', color='gray'),
                fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Gradient comparison
    ax = axes[1]
    
    # MSE gradient: -2(1-p) 
    mse_grad = np.abs(-2 * (1 - p))
    
    # CE gradient: -1/p (before sigmoid) -> simplified for viz
    ce_grad = np.abs(1 / p)
    ce_grad = np.clip(ce_grad, 0, 10)  # Clip for visualization
    
    ax.plot(p, mse_grad, color=COLORS['secondary'], linewidth=2.5, label='|∂MSE/∂ŷ| = 2|y-ŷ|')
    ax.plot(p, ce_grad, color=COLORS['primary'], linewidth=2.5, label='|∂CE/∂ŷ| = 1/ŷ')
    ax.set_xlabel('Predicted Probability ŷ', fontsize=12)
    ax.set_ylabel('|Gradient|', fontsize=12)
    ax.set_title('Gradient Magnitude Comparison', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 10)
    
    ax.text(0.5, 0.95, 'CE gradient is LARGE when\nprediction is confident & wrong',
            transform=ax.transAxes, fontsize=9, verticalalignment='top', horizontalalignment='center',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.suptitle('Why Cross-Entropy is Better for Classification', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('cross_entropy_vs_mse.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Generated cross_entropy_vs_mse.png")


def main():
    """Generate all figures."""
    print("Generating figures for ML Interview Guide...\n")
    
    generate_sigmoid_figure()
    generate_activation_functions()
    generate_loss_landscape()
    generate_learning_curves()
    generate_bias_variance()
    generate_softmax_temperature()
    generate_attention_heatmap()
    generate_lr_schedules()
    generate_gradient_flow()
    generate_overfitting_spectrum()
    generate_learning_curves_diagnostic()
    generate_cross_entropy_vs_mse()
    
    print("\n✅ All figures generated successfully!")
    print("Figures are saved in the current directory (ml-prep/figures/)")


if __name__ == "__main__":
    main()
