"""
Generate visualization comparing Laplace and Gaussian priors for MAP estimation.
Shows why Laplace → L1 (sparsity) and Gaussian → L2 (weight decay).
"""

import numpy as np
import matplotlib.pyplot as plt

# Define distributions manually (no scipy dependency)
def gaussian_pdf(x, mu=0, sigma=1):
    """Gaussian probability density function."""
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def laplace_pdf_func(x, mu=0, b=1):
    """Laplace probability density function."""
    return (1 / (2 * b)) * np.exp(-np.abs(x - mu) / b)

# Set style - no LaTeX to avoid multiline text issues
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'cm'

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# =============================================================================
# Panel 1: Prior Distributions (PDFs)
# =============================================================================
ax1 = axes[0, 0]

theta = np.linspace(-5, 5, 500)

# Gaussian prior: N(0, 1)
gaussian_vals = gaussian_pdf(theta, mu=0, sigma=1)

# Laplace prior: Laplace(0, 1) - scale chosen so variance matches Gaussian
# Laplace variance = 2b², so b = 1/√2 ≈ 0.707 for variance = 1
laplace_scale = 1 / np.sqrt(2)
laplace_vals = laplace_pdf_func(theta, mu=0, b=laplace_scale)

ax1.plot(theta, gaussian_vals, 'b-', linewidth=2.5, label=r'Gaussian: $P(\theta) \propto e^{-\frac{\theta^2}{2\sigma^2}}$')
ax1.plot(theta, laplace_vals, 'r-', linewidth=2.5, label=r'Laplace: $P(\theta) \propto e^{-\frac{|\theta|}{b}}$')
ax1.fill_between(theta, 0, gaussian_vals, alpha=0.2, color='blue')
ax1.fill_between(theta, 0, laplace_vals, alpha=0.2, color='red')

# Annotate key differences
ax1.annotate('Sharp peak\n(many zeros)', xy=(0, laplace_vals[250]), xytext=(1.5, 0.9),
            fontsize=10, color='indianred',
            arrowprops=dict(arrowstyle='->', color='lightcoral', lw=0.8, linestyle='--'),
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='lightcoral', alpha=0.9))

ax1.annotate('Smooth peak\n(small values)', xy=(0, gaussian_vals[250]), xytext=(-2.5, 0.55),
            fontsize=10, color='steelblue',
            arrowprops=dict(arrowstyle='->', color='cornflowerblue', lw=0.8, linestyle='--'),
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='cornflowerblue', alpha=0.9))

# Heavy tails annotation - point to where the difference is more visible
# At x=2, Laplace has higher probability than Gaussian (heavier tail)
laplace_at_2 = laplace_pdf_func(2, mu=0, b=laplace_scale)
gaussian_at_2 = gaussian_pdf(2, mu=0, sigma=1)
ax1.annotate(f'Heavier tails\nLaplace: {laplace_at_2:.3f}\nGaussian: {gaussian_at_2:.3f}', 
            xy=(2, laplace_at_2), 
            xytext=(3.2, 0.25),
            fontsize=9, color='indianred',
            arrowprops=dict(arrowstyle='->', color='lightcoral', lw=0.8, linestyle='--'),
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='lightcoral', alpha=0.8))

ax1.set_xlabel(r'Parameter $\theta$')
ax1.set_ylabel(r'Prior probability $P(\theta)$')
ax1.set_title('Prior Distributions: Laplace vs Gaussian', fontweight='bold')
ax1.legend(loc='upper left', fontsize=10)
ax1.set_xlim(-5, 5)
ax1.set_ylim(0, 1.1)
ax1.grid(True, alpha=0.3)

# =============================================================================
# Panel 2: Log-Prior (What Optimization Sees)
# =============================================================================
ax2 = axes[0, 1]

theta_positive = np.linspace(-3, 3, 500)

# Log of Gaussian: -θ²/2σ² (parabola)
log_gaussian = -theta_positive**2 / 2

# Log of Laplace: -|θ|/b (V-shape)
log_laplace = -np.abs(theta_positive) / laplace_scale

# Normalize so they're on same scale for visualization
log_gaussian_norm = log_gaussian / np.max(np.abs(log_gaussian)) * 3
log_laplace_norm = log_laplace / np.max(np.abs(log_laplace)) * 3

ax2.plot(theta_positive, log_gaussian_norm, 'b-', linewidth=2.5, 
         label=r'$\log P_{\text{Gaussian}}(\theta) \propto -\theta^2$ (L2)')
ax2.plot(theta_positive, log_laplace_norm, 'r-', linewidth=2.5, 
         label=r'$\log P_{\text{Laplace}}(\theta) \propto -|\theta|$ (L1)')

# Annotate the key insight - centered in the figure
ax2.text(0.5, 0.5, 'Key Insight:\nlog P(θ) becomes\npenalty term!',
        transform=ax2.transAxes, fontsize=9, va='center', ha='center',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, edgecolor='orange'))

# Show the penalty type
ax2.annotate(r'Quadratic penalty', xy=(2, log_gaussian_norm[int(500*5/6)]), xytext=(2.3, -1),
            fontsize=10, color='steelblue',
            arrowprops=dict(arrowstyle='->', color='cornflowerblue', lw=0.8, linestyle='--'))

ax2.annotate('Linear penalty\n(non-differentiable at 0)', xy=(0, 0), xytext=(-2.5, -0.5),
            fontsize=10, color='indianred',
            arrowprops=dict(arrowstyle='->', color='lightcoral', lw=0.8, linestyle='--'))

ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax2.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

ax2.set_xlabel(r'Parameter $\theta$')
ax2.set_ylabel(r'$\log P(\theta)$ (negative = penalty)')
ax2.set_title('Log-Prior: Why We Get L1 vs L2 Regularization', fontweight='bold')
ax2.legend(loc='lower left', fontsize=9)
ax2.set_xlim(-3, 3)
ax2.grid(True, alpha=0.3)

# =============================================================================
# Panel 3: Effect on Weights After Training
# =============================================================================
ax3 = axes[1, 0]

np.random.seed(42)

# Simulate trained weights
n_weights = 50

# L2 regularization: all weights small but non-zero (Gaussian-like)
l2_weights = np.random.normal(0, 0.3, n_weights)

# L1 regularization: many zeros, some larger weights (sparse)
l1_weights = np.zeros(n_weights)
n_nonzero = 12  # Only 12 out of 50 are non-zero
nonzero_indices = np.random.choice(n_weights, n_nonzero, replace=False)
l1_weights[nonzero_indices] = np.random.laplace(0, 0.4, n_nonzero)

# Plot as bar charts
width = 0.35
x = np.arange(n_weights)

bars1 = ax3.bar(x - width/2, np.abs(l2_weights), width, label=r'L2 (Gaussian prior)', 
               color='blue', alpha=0.7, edgecolor='darkblue')
bars2 = ax3.bar(x + width/2, np.abs(l1_weights), width, label=r'L1 (Laplace prior)', 
               color='red', alpha=0.7, edgecolor='darkred')

ax3.set_xlabel('Weight index')
ax3.set_ylabel(r'$|w_i|$ (absolute weight magnitude)')
ax3.set_title('Effect on Weights: L1 Creates Sparsity', fontweight='bold')
ax3.legend(loc='upper right', fontsize=9)
ax3.set_xlim(-1, 50)

# Add annotation boxes - shifted to avoid legend overlap
ax3.text(0.12, 0.80, 
         'L2 (Gaussian):\n' +
         'All weights small\n' +
         'None exactly zero\n' +
         '→ Weight decay',
         transform=ax3.transAxes, fontsize=9, va='top', ha='center',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9, edgecolor='lightsteelblue'))

ax3.text(0.70, 0.80, 
         'L1 (Laplace):\n' +
         'Many exact zeros\n' +
         'Few non-zero\n' +
         '→ Sparsity/Selection',
         transform=ax3.transAxes, fontsize=9, va='top', ha='center',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, edgecolor='wheat'))

# Count zeros annotation
n_zeros_l1 = np.sum(np.abs(l1_weights) < 1e-10)
ax3.text(0.5, 0.5, f'L1: {n_zeros_l1}/50 weights = 0\nL2: 0/50 weights = 0',
        transform=ax3.transAxes, fontsize=10, ha='center',
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.9))

ax3.grid(True, alpha=0.3, axis='y')

# =============================================================================
# Panel 4: Geometric Intuition (Constraint Regions)
# =============================================================================
ax4 = axes[1, 1]

# Create L1 ball (diamond) and L2 ball (circle)
t = np.linspace(0, 2*np.pi, 100)

# L2 ball (circle): w1² + w2² ≤ 1
r = 1
l2_w1 = r * np.cos(t)
l2_w2 = r * np.sin(t)

# L1 ball (diamond): |w1| + |w2| ≤ 1
l1_w1 = np.array([1, 0, -1, 0, 1])
l1_w2 = np.array([0, 1, 0, -1, 0])

ax4.fill(l2_w1, l2_w2, alpha=0.3, color='blue', label=r'L2 ball: $\|w\|_2^2 \leq c$')
ax4.plot(l2_w1, l2_w2, 'b-', linewidth=2)

ax4.fill(l1_w1, l1_w2, alpha=0.3, color='red', label=r'L1 ball: $\|w\|_1 \leq c$')
ax4.plot(l1_w1, l1_w2, 'r-', linewidth=2)

# Add loss contours (ellipses representing optimization objective)
from matplotlib.patches import Ellipse
for i, (a, b) in enumerate([(1.5, 0.8), (2.0, 1.1), (2.5, 1.4)]):
    ellipse = Ellipse((1.2, 0.8), width=a, height=b, angle=-30,
                      fill=False, color='green', linestyle='--', linewidth=1.5, alpha=0.7-i*0.2)
    ax4.add_patch(ellipse)

# Mark where solutions hit
ax4.scatter([1, 0], [0, 1], color='red', s=100, zorder=5, marker='o')  # L1 corners
ax4.scatter([0.6], [0.8], color='blue', s=100, zorder=5, marker='s')   # L2 smooth point

# Annotations
ax4.annotate('L1 solution\n(sparse: $w_2=0$)', xy=(1, 0), xytext=(1.3, -0.5),
            fontsize=10, color='indianred',
            arrowprops=dict(arrowstyle='->', color='lightcoral', lw=0.8, linestyle='--'),
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='lightcoral', alpha=0.9))

ax4.annotate('L2 solution\n(both non-zero)', xy=(0.6, 0.8), xytext=(0.2, 1.5),
            fontsize=10, color='steelblue',
            arrowprops=dict(arrowstyle='->', color='cornflowerblue', lw=0.8, linestyle='--'),
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='cornflowerblue', alpha=0.9))

ax4.annotate('Loss contours', xy=(1.8, 0.3), fontsize=9, color='green')

# Corners annotation - moved slightly up
ax4.text(0.5, 0.05, 
         'Key: L1 diamond has corners on axes\n' +
         '→ Solutions often hit corners → exact zeros!',
         transform=ax4.transAxes, fontsize=9, ha='center',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, edgecolor='orange'))

ax4.set_xlabel(r'$w_1$')
ax4.set_ylabel(r'$w_2$')
ax4.set_title('Geometric View: Why L1 Induces Sparsity', fontweight='bold')
ax4.legend(loc='upper left', fontsize=9)
ax4.set_xlim(-1.8, 2.2)
ax4.set_ylim(-1.5, 1.8)
ax4.set_aspect('equal')
ax4.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
ax4.axvline(x=0, color='gray', linestyle='-', linewidth=0.5)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ml-notes/figures/laplace_gaussian_prior.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()

print("Generated: ml-notes/figures/laplace_gaussian_prior.png")
