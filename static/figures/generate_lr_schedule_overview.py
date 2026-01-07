"""
Generate learning rate schedule overview visualization.
Shows different LR schedules: constant, step decay, cosine, warmup+cosine.
"""

import matplotlib.pyplot as plt
import numpy as np

# Set up the figure
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

total_steps = 1000
steps = np.arange(total_steps)
base_lr = 0.001
warmup_steps = 100

# Top left: Constant vs Step Decay
ax1 = axes[0, 0]
constant_lr = np.ones(total_steps) * base_lr
step_decay_lr = np.where(steps < 300, base_lr, 
                 np.where(steps < 600, base_lr * 0.1, base_lr * 0.01))

ax1.plot(steps, constant_lr * 1000, 'b-', linewidth=2, label='Constant LR')
ax1.plot(steps, step_decay_lr * 1000, 'r-', linewidth=2, label='Step Decay (×0.1 at 300, 600)')
ax1.set_xlabel('Training Step', fontsize=11)
ax1.set_ylabel('Learning Rate (×10⁻³)', fontsize=11)
ax1.set_title('Constant vs Step Decay', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 1.2)

# Top right: Cosine Annealing
ax2 = axes[0, 1]
cosine_lr = base_lr * 0.5 * (1 + np.cos(np.pi * steps / total_steps))

ax2.plot(steps, cosine_lr * 1000, 'g-', linewidth=2, label='Cosine Annealing')
ax2.fill_between(steps, 0, cosine_lr * 1000, alpha=0.3, color='green')
ax2.set_xlabel('Training Step', fontsize=11)
ax2.set_ylabel('Learning Rate (×10⁻³)', fontsize=11)
ax2.set_title('Cosine Annealing', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0, 1.2)

# Bottom left: Warmup + Cosine (common for transformers)
ax3 = axes[1, 0]
warmup_cosine_lr = np.where(
    steps < warmup_steps,
    base_lr * steps / warmup_steps,  # Linear warmup
    base_lr * 0.5 * (1 + np.cos(np.pi * (steps - warmup_steps) / (total_steps - warmup_steps)))  # Cosine decay
)

ax3.plot(steps, warmup_cosine_lr * 1000, 'purple', linewidth=2, label='Warmup + Cosine')
ax3.axvline(x=warmup_steps, color='gray', linestyle='--', alpha=0.5, label=f'Warmup ends (step {warmup_steps})')
ax3.fill_between(steps, 0, warmup_cosine_lr * 1000, alpha=0.3, color='purple')
ax3.set_xlabel('Training Step', fontsize=11)
ax3.set_ylabel('Learning Rate (×10⁻³)', fontsize=11)
ax3.set_title('Warmup + Cosine Decay (Transformer Default)', fontsize=12, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.set_ylim(0, 1.2)

# Bottom right: Explanation
ax4 = axes[1, 1]
ax4.axis('off')

explanation_text = """
┌──────────────────────────────────────────────────────────────────────┐
│                 LEARNING RATE SCHEDULES                              │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  WHY WARMUP?                                                         │
│  • Adam's moment estimates (m_t, v_t) are unreliable early on       │
│  • Large LR + bad estimates = unstable training                      │
│  • Warmup lets estimates stabilize before large updates              │
│                                                                      │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  WHY DECAY?                                                          │
│  • Large LR early: Fast progress, escape local minima                │
│  • Small LR late: Fine-tune, converge to precise minimum             │
│  • Cosine is smooth; step decay is simpler but has jumps             │
│                                                                      │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  FORMULAS                                                            │
│                                                                      │
│  Linear Warmup:    lr(t) = lr_max × (t / warmup_steps)              │
│                                                                      │
│  Cosine Decay:     lr(t) = lr_max × 0.5 × (1 + cos(πt/T))           │
│                                                                      │
│  Step Decay:       lr(t) = lr_initial × γ^(floor(t / step_size))    │
│                                                                      │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  TYPICAL CHOICES                                                     │
│  • LLMs/Transformers: Warmup (1-10% of training) + Cosine decay     │
│  • CNNs: Step decay (÷10 at 30%, 60%, 90% of training)              │
│  • Fine-tuning: Small constant LR or linear decay                    │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
"""

ax4.text(0.02, 0.95, explanation_text, transform=ax4.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='#E8F5E9', edgecolor='#4CAF50', alpha=0.9))

plt.suptitle('Learning Rate Schedules: Why and When', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('lr_schedule_overview.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()

print("Generated lr_schedule_overview.png")
