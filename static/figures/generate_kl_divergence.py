#!/usr/bin/env python3
"""Generate KL divergence visualization: Forward KL vs Reverse KL on bimodal distribution."""

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.size'] = 12

# Gaussian PDF using numpy only
def gaussian_pdf(x, mu, sigma):
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

# Create figure with three subplots
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4))

# Generate x values
x = np.linspace(-6, 6, 500)

# True bimodal distribution P (mixture of two Gaussians)
def bimodal_p(x):
    return 0.5 * gaussian_pdf(x, -2, 0.8) + 0.5 * gaussian_pdf(x, 2, 0.8)

# Forward KL result Q (mode-covering: wide Gaussian)
def forward_kl_q(x):
    return gaussian_pdf(x, 0, 2.5)

# Reverse KL result Q (mode-seeking: narrow Gaussian at one mode)
def reverse_kl_q(x):
    return gaussian_pdf(x, -2, 0.8)

p = bimodal_p(x)
q_forward = forward_kl_q(x)
q_reverse = reverse_kl_q(x)

# ===== Panel 1: True Distribution P (Bimodal) =====
ax1.fill_between(x, p, alpha=0.4, color='#2563eb')
ax1.plot(x, p, color='#2563eb', linewidth=2.5, label='P (true)')
ax1.axvline(x=-2, color='#2563eb', linestyle='--', alpha=0.5)
ax1.axvline(x=2, color='#2563eb', linestyle='--', alpha=0.5)
ax1.set_xlabel('x', fontsize=12)
ax1.set_ylabel('Probability Density', fontsize=12)
ax1.set_title('True Distribution P\n(Bimodal)', fontsize=14, fontweight='bold')
ax1.set_xlim(-6, 6)
ax1.set_ylim(0, 0.35)
ax1.legend(loc='upper right', fontsize=10)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.grid(True, alpha=0.3)

# ===== Panel 2: Forward KL Result (Mode-Covering) =====
ax2.fill_between(x, p, alpha=0.3, color='#2563eb')
ax2.plot(x, p, color='#2563eb', linewidth=2, label='P (true)', alpha=0.7)
ax2.fill_between(x, q_forward, alpha=0.3, color='#dc2626')
ax2.plot(x, q_forward, color='#dc2626', linewidth=2.5, label='Q (forward KL)', linestyle='--')

# Annotate
ax2.annotate('Q covers\nboth modes', xy=(0, 0.17), fontsize=10, ha='center', 
             color='#dc2626', fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

ax2.set_xlabel('x', fontsize=12)
ax2.set_ylabel('Probability Density', fontsize=12)
ax2.set_title('Forward KL: D(P||Q)\n(Mode-Covering)', fontsize=14, fontweight='bold', color='#dc2626')
ax2.set_xlim(-6, 6)
ax2.set_ylim(0, 0.35)
ax2.legend(loc='upper right', fontsize=9)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.grid(True, alpha=0.3)

# ===== Panel 3: Reverse KL Result (Mode-Seeking) =====
ax3.fill_between(x, p, alpha=0.3, color='#2563eb')
ax3.plot(x, p, color='#2563eb', linewidth=2, label='P (true)', alpha=0.7)
ax3.fill_between(x, q_reverse, alpha=0.4, color='#16a34a')
ax3.plot(x, q_reverse, color='#16a34a', linewidth=2.5, label='Q (reverse KL)', linestyle='--')

# Annotate
ax3.annotate('Q picks\none mode', xy=(-2, 0.28), fontsize=10, ha='center', 
             color='#16a34a', fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
ax3.annotate('Ignores\nthis mode', xy=(2, 0.15), fontsize=9, ha='center', 
             color='gray', style='italic')

ax3.set_xlabel('x', fontsize=12)
ax3.set_ylabel('Probability Density', fontsize=12)
ax3.set_title('Reverse KL: D(Q||P)\n(Mode-Seeking)', fontsize=14, fontweight='bold', color='#16a34a')
ax3.set_xlim(-6, 6)
ax3.set_ylim(0, 0.35)
ax3.legend(loc='upper right', fontsize=9)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.grid(True, alpha=0.3)

# Add summary text at bottom
fig.text(0.5, 0.01, 
         'Forward KL penalizes Q=0 where P>0 → covers all modes  |  '
         'Reverse KL penalizes P=0 where Q>0 → picks one mode',
         ha='center', fontsize=10, style='italic')

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig('kl_divergence.png', bbox_inches='tight', facecolor='white')
plt.close()

print("Generated kl_divergence.png")
