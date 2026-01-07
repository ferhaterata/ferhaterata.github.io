"""
Generate Adam optimizer intuition visualization.
Shows how Adam adapts learning rate for different parameters.
"""

import matplotlib.pyplot as plt
import numpy as np

# Set up the figure
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Top left: Two parameters with different gradient behaviors
ax1 = axes[0, 0]
np.random.seed(42)
steps = np.arange(50)

# Parameter 1: Large, noisy gradients
grad1 = np.random.randn(50) * 5 + np.sin(steps * 0.3) * 3

# Parameter 2: Small, consistent gradients
grad2 = np.random.randn(50) * 0.5 + 1

ax1.plot(steps, grad1, 'b-', alpha=0.7, linewidth=2, label='Param 1: Large, noisy gradients')
ax1.plot(steps, grad2, 'r-', alpha=0.7, linewidth=2, label='Param 2: Small, consistent gradients')
ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax1.set_xlabel('Training Step', fontsize=11)
ax1.set_ylabel('Gradient Value', fontsize=11)
ax1.set_title('Gradient Behavior for Two Parameters', fontsize=12, fontweight='bold')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

# Top right: Accumulated v_t (second moment) for Adam
ax2 = axes[0, 1]
beta2 = 0.999
v1 = np.zeros(50)
v2 = np.zeros(50)
v1[0] = grad1[0]**2
v2[0] = grad2[0]**2
for t in range(1, 50):
    v1[t] = beta2 * v1[t-1] + (1 - beta2) * grad1[t]**2
    v2[t] = beta2 * v2[t-1] + (1 - beta2) * grad2[t]**2

ax2.plot(steps, np.sqrt(v1), 'b-', linewidth=2, label='√v₁ (Param 1)')
ax2.plot(steps, np.sqrt(v2), 'r-', linewidth=2, label='√v₂ (Param 2)')
ax2.set_xlabel('Training Step', fontsize=11)
ax2.set_ylabel('√v_t (RMS of gradients)', fontsize=11)
ax2.set_title('Second Moment Estimate (√v_t)', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Bottom left: Effective learning rate
ax3 = axes[1, 0]
base_lr = 0.001
epsilon = 1e-8
eff_lr1 = base_lr / (np.sqrt(v1) + epsilon)
eff_lr2 = base_lr / (np.sqrt(v2) + epsilon)

ax3.plot(steps, eff_lr1 * 1000, 'b-', linewidth=2, label='Param 1: eff. LR (×1000)')
ax3.plot(steps, eff_lr2 * 1000, 'r-', linewidth=2, label='Param 2: eff. LR (×1000)')
ax3.axhline(y=base_lr * 1000, color='k', linestyle='--', alpha=0.5, label='Base LR')
ax3.set_xlabel('Training Step', fontsize=11)
ax3.set_ylabel('Effective Learning Rate (×1000)', fontsize=11)
ax3.set_title('Adaptive Learning Rate per Parameter', fontsize=12, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Bottom right: Summary table as text
ax4 = axes[1, 1]
ax4.axis('off')

summary_text = """
┌─────────────────────────────────────────────────────────────────┐
│                    ADAM INTUITION SUMMARY                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Parameter with LARGE, NOISY gradients:                         │
│  • v_t accumulates large values → √v_t is large                 │
│  • Effective LR = base_lr / √v_t → SMALLER steps                │
│  • Prevents overshooting on parameters with high variance       │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Parameter with SMALL, CONSISTENT gradients:                    │
│  • v_t accumulates small values → √v_t is small                 │
│  • Effective LR = base_lr / √v_t → LARGER steps                 │
│  • Accelerates learning on parameters that need it              │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  KEY FORMULA:  θ_new = θ - lr × m_t / (√v_t + ε)                │
│                                                                 │
│  m_t = momentum (direction)                                     │
│  v_t = squared gradient history (scale/trust)                   │
│  Dividing by √v_t "normalizes" the update magnitude             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
"""

ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='#E3F2FD', edgecolor='#2196F3', alpha=0.9))

plt.suptitle('How Adam Adapts Learning Rates to Parameter Gradient Statistics', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('adam_intuition.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()

print("Generated adam_intuition.png")
