"""
Generate variance propagation visualization for weight initialization.
Shows why He/Xavier initialization values are what they are.
"""

import matplotlib.pyplot as plt
import numpy as np

# Set up the figure
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

np.random.seed(42)

# Simulate variance propagation through layers
def simulate_forward_pass(n_layers, n_neurons, init_std, activation='relu'):
    """Simulate forward pass and track activation statistics."""
    x = np.random.randn(1000, n_neurons)  # Input batch
    variances = [np.var(x)]
    
    for _ in range(n_layers):
        W = np.random.randn(n_neurons, n_neurons) * init_std
        x = x @ W
        
        if activation == 'relu':
            x = np.maximum(0, x)
        elif activation == 'tanh':
            x = np.tanh(x)
        # linear: no activation
        
        variances.append(np.var(x))
    
    return variances

# Top left: Bad init (too small) - vanishing
ax1 = axes[0, 0]
n_layers = 10
n_neurons = 256

vars_small = simulate_forward_pass(n_layers, n_neurons, init_std=0.01, activation='relu')
vars_large = simulate_forward_pass(n_layers, n_neurons, init_std=1.0, activation='relu')
vars_he = simulate_forward_pass(n_layers, n_neurons, init_std=np.sqrt(2/n_neurons), activation='relu')

layers = np.arange(len(vars_small))
ax1.semilogy(layers, vars_small, 'b-o', linewidth=2, markersize=6, label='Too small (std=0.01)')
ax1.semilogy(layers, vars_large, 'r-o', linewidth=2, markersize=6, label='Too large (std=1.0)')
ax1.semilogy(layers, vars_he, 'g-o', linewidth=2, markersize=6, label='He init (std=√(2/n))')
ax1.axhline(y=1, color='k', linestyle='--', alpha=0.5, label='Target variance')
ax1.set_xlabel('Layer', fontsize=11)
ax1.set_ylabel('Activation Variance (log scale)', fontsize=11)
ax1.set_title('Variance Propagation with ReLU', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Top right: Xavier for tanh
ax2 = axes[0, 1]
vars_small_tanh = simulate_forward_pass(n_layers, n_neurons, init_std=0.01, activation='tanh')
vars_large_tanh = simulate_forward_pass(n_layers, n_neurons, init_std=1.0, activation='tanh')
vars_xavier = simulate_forward_pass(n_layers, n_neurons, init_std=np.sqrt(2/(n_neurons + n_neurons)), activation='tanh')

ax2.semilogy(layers, vars_small_tanh, 'b-o', linewidth=2, markersize=6, label='Too small (std=0.01)')
ax2.semilogy(layers, vars_large_tanh, 'r-o', linewidth=2, markersize=6, label='Too large (std=1.0)')
ax2.semilogy(layers, vars_xavier, 'g-o', linewidth=2, markersize=6, label='Xavier (std=√(2/(n_in+n_out)))')
ax2.axhline(y=1, color='k', linestyle='--', alpha=0.5, label='Target variance')
ax2.set_xlabel('Layer', fontsize=11)
ax2.set_ylabel('Activation Variance (log scale)', fontsize=11)
ax2.set_title('Variance Propagation with Tanh', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Bottom left: Derivation explanation
ax3 = axes[1, 0]
ax3.axis('off')

derivation_text = """
┌──────────────────────────────────────────────────────────────────────┐
│                    VARIANCE DERIVATION                               │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  For a linear layer y = Wx where x has n_in elements:                │
│                                                                      │
│    y_j = Σᵢ W_ji × x_i                                               │
│                                                                      │
│  Assuming W and x are independent with zero mean:                    │
│                                                                      │
│    Var(y_j) = Σᵢ Var(W_ji × x_i)                                     │
│             = Σᵢ Var(W_ji) × Var(x_i)                                │
│             = n_in × Var(W) × Var(x)                                 │
│                                                                      │
│  To keep Var(y) = Var(x), we need:                                   │
│                                                                      │
│    n_in × Var(W) = 1  →  Var(W) = 1/n_in                             │
│                                                                      │
│  For ReLU: ReLU kills ~half the values, so:                          │
│    Var(ReLU(y)) ≈ Var(y)/2                                           │
│                                                                      │
│  To compensate: Var(W) = 2/n_in  →  std = √(2/n_in)  [He Init]       │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
"""

ax3.text(0.02, 0.95, derivation_text, transform=ax3.transAxes, fontsize=10,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='#FFF9C4', edgecolor='#FFC107', alpha=0.9))

# Bottom right: Summary table
ax4 = axes[1, 1]
ax4.axis('off')

summary_text = """
┌──────────────────────────────────────────────────────────────────────┐
│                 INITIALIZATION SUMMARY                               │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────┬───────────────────┬────────────────────────────┐    │
│  │ Activation  │ Init Method       │ Variance                   │    │
│  ├─────────────┼───────────────────┼────────────────────────────┤    │
│  │ Linear      │ Xavier/Glorot     │ 2 / (n_in + n_out)         │    │
│  │ Sigmoid     │ Xavier/Glorot     │ 2 / (n_in + n_out)         │    │
│  │ Tanh        │ Xavier/Glorot     │ 2 / (n_in + n_out)         │    │
│  │ ReLU        │ He/Kaiming        │ 2 / n_in                   │    │
│  │ Leaky ReLU  │ He (modified)     │ 2 / (1 + α²) / n_in        │    │
│  └─────────────┴───────────────────┴────────────────────────────┘    │
│                                                                      │
│  Key Insight:                                                        │
│  • Xavier: Balances forward AND backward variance propagation        │
│  • He: Compensates for ReLU zeroing half the activations             │
│  • Goal: Var(output) ≈ Var(input) at every layer                     │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
"""

ax4.text(0.02, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='#E8F5E9', edgecolor='#4CAF50', alpha=0.9))

plt.suptitle('Why He and Xavier Initialization Values?', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('variance_propagation.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()

print("Generated variance_propagation.png")
