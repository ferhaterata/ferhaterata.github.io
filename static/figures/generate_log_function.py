"""
Generate visualization of the log function and why we use log-likelihood.
"""

import numpy as np
import matplotlib.pyplot as plt

# Set style - Enable full LaTeX rendering for publication-quality math
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['text.usetex'] = True  # Use full LaTeX rendering
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Computer Modern Roman']
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}\usepackage{amssymb}'  # For \text{} and \checkmark

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# =============================================================================
# Panel 1: The Log Function (Monotonically Increasing)
# =============================================================================
ax1 = axes[0, 0]
x = np.linspace(0.01, 5, 500)
y = np.log(x)

ax1.plot(x, y, 'b-', linewidth=2.5, label=r'$y = \log(x)$')
ax1.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
ax1.axvline(x=1, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

# Annotate key points - simple text labels, no arrows
ax1.scatter([1], [0], color='red', s=80, zorder=5)
ax1.text(1.15, -0.4, r'$\log(1) = 0$', fontsize=10, color='red',
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='red', alpha=0.9))

ax1.scatter([np.e], [1], color='green', s=80, zorder=5)
ax1.text(np.e + 0.3, 0.5, r'$\log(e) = 1$', fontsize=10, color='green',
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='green', alpha=0.9))

# Show monotonically increasing property - simple text box, no arrow
ax1.text(0.3, 1.3, 'Monotonically\nIncreasing', fontsize=10, color='purple',
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='purple', alpha=0.8))

ax1.set_xlabel('x')
ax1.set_ylabel(r'$\log(x)$')
ax1.set_title(r'The Natural Log Function' + '\n' + r'(Preserves argmax: if $a > b$, then $\log(a) > \log(b)$)')
ax1.set_xlim(0, 5)
ax1.set_ylim(-3, 2)
ax1.grid(True, alpha=0.3)
ax1.legend(loc='lower right')

# =============================================================================
# Panel 2: Log Transforms Products to Sums (using LaTeX mathtext)
# =============================================================================
ax2 = axes[0, 1]
ax2.axis('off')

# Create structured text with proper LaTeX math rendering
y_pos = 0.95
line_height = 0.085  # Increased spacing between formula lines

ax2.set_title(r'Why Products $\rightarrow$ Sums Helps', fontsize=12, fontweight='bold')

y_pos -= 0.08  # Add margin at top for yellow box

# Likelihood section
ax2.text(0.05, y_pos, r'$\mathbf{Likelihood:}$', transform=ax2.transAxes, fontsize=11)
y_pos -= line_height
ax2.text(0.1, y_pos, r'$L(\theta) = P(x_1|\theta) \times P(x_2|\theta) \times \cdots \times P(x_N|\theta)$', 
        transform=ax2.transAxes, fontsize=11)
y_pos -= line_height
ax2.text(0.2, y_pos, r'$= \prod_{i=1}^{N} P(x_i|\theta)$', transform=ax2.transAxes, fontsize=11)

y_pos -= 0.11  # More space between sections

# Log-Likelihood section
ax2.text(0.05, y_pos, r'$\mathbf{Log\text{-}Likelihood:}$', transform=ax2.transAxes, fontsize=11)
y_pos -= line_height
ax2.text(0.1, y_pos, r'$\ell(\theta) = \log P(x_1|\theta) + \log P(x_2|\theta) + \cdots + \log P(x_N|\theta)$', 
        transform=ax2.transAxes, fontsize=11)
y_pos -= line_height
ax2.text(0.2, y_pos, r'$= \sum_{i=1}^{N} \log P(x_i|\theta)$', transform=ax2.transAxes, fontsize=11)

y_pos -= 0.11  # More space between sections

# Derivative section
ax2.text(0.05, y_pos, r'$\mathbf{Derivative:}$', transform=ax2.transAxes, fontsize=11)
y_pos -= line_height
ax2.text(0.1, y_pos, r'$\frac{d}{d\theta}\left[\prod_i f_i\right] = \mathrm{messy\ (product\ rule)}$', 
        transform=ax2.transAxes, fontsize=11, color='red')
y_pos -= line_height * 1.2
ax2.text(0.1, y_pos, r'$\frac{d}{d\theta}\left[\sum_i \log f_i\right] = \sum_i \frac{1}{f_i} \cdot \frac{df_i}{d\theta}$  $\checkmark$ clean!', 
        transform=ax2.transAxes, fontsize=11, color='green')

# Add a box around the content (adjusted to surround all formulas with margin)
ax2.add_patch(plt.Rectangle((0.02, 0.04), 0.96, 0.92, transform=ax2.transAxes,
              fill=True, facecolor='lightyellow', edgecolor='orange', 
              linewidth=2, alpha=0.3))

# =============================================================================
# Panel 3: Numerical Stability - Show ACTUAL numbers underflowing
# =============================================================================
ax3 = axes[1, 0]

# Create a bar chart showing the actual computation
n_samples_list = [10, 50, 100, 200, 324]
prob = 0.1  # Each sample has probability 0.1

# Compute raw likelihood and log-likelihood
raw_likelihoods = []
log_likelihoods = []
for n in n_samples_list:
    raw_likelihoods.append(prob ** n)
    log_likelihoods.append(n * np.log(prob))

# Create side-by-side bars
x_pos = np.arange(len(n_samples_list))
width = 0.35

# For raw likelihood, use a different representation since values underflow
# Show the log10 of what the number SHOULD be
raw_likelihood_log10 = [n * np.log10(prob) for n in n_samples_list]

bars1 = ax3.bar(x_pos - width/2, raw_likelihood_log10, width, label=r'$\log_{10}$(Raw Likelihood)', 
               color='red', alpha=0.7)
bars2 = ax3.bar(x_pos + width/2, log_likelihoods, width, label=r'Log-Likelihood (computable)', 
               color='green', alpha=0.7)

# Add value labels on bars
for i, (bar, n, raw) in enumerate(zip(bars1, n_samples_list, raw_likelihoods)):
    if raw == 0:
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 30, 
                f'UNDERFLOW!\n(= 0 in float64)', ha='center', va='top', fontsize=8, 
                color='red', fontweight='bold')
    elif raw < 1e-300:
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 30,
                r'$\approx 0$' + '\n(too small)', ha='center', va='top', fontsize=8, color='red')
    else:
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 30,
                f'{raw:.1e}', ha='center', va='top', fontsize=8, color='red')

for bar, val in zip(bars2, log_likelihoods):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
            f'{val:.1f}', ha='center', va='bottom', fontsize=8, color='green')

ax3.set_xticks(x_pos)
ax3.set_xticklabels([f'n={n}' for n in n_samples_list])
ax3.set_ylabel('Value (log scale for comparison)')
ax3.set_title('Numerical Stability: Raw Likelihood Underflows!\n' + 
              r'(Each sample has $P(x) = 0.1$, computing $0.1^n$)')
ax3.legend(loc='lower left', bbox_to_anchor=(0.02, 0.15))
ax3.axhline(y=-308, color='red', linestyle='--', alpha=0.5, linewidth=2)
ax3.text(0.5, -280, r'float64 limit $\approx 10^{-308}$', fontsize=9, color='red', ha='left')

# Add annotation box explaining the problem
ax3.text(0.02, 0.02, 
         r'Problem: $0.1^{324} \approx 10^{-324} < 10^{-308} \rightarrow$ Rounds to 0!' + '\n' +
         r'Solution: $\log(0.1^{324}) = 324 \times \log(0.1) \approx -746$ $\checkmark$',
         transform=ax3.transAxes, fontsize=9, va='bottom',
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8, edgecolor='black'))

# =============================================================================
# Panel 4: Why Log "Expands" Small Values - The Derivative Perspective
# =============================================================================
ax4 = axes[1, 1]

# The key insight: d/dx log(x) = 1/x
# Small x -> large gradient (more sensitive to changes)
# Large x -> small gradient (less sensitive to changes)

x_range = np.linspace(0.05, 3, 100)
derivative = 1 / x_range  # d/dx log(x) = 1/x

ax4.plot(x_range, derivative, 'b-', linewidth=2.5, label=r'$\frac{d}{dx}\log(x) = \frac{1}{x}$')
ax4.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
ax4.axvline(x=1, color='gray', linestyle='--', alpha=0.5)

# Highlight the two regions
ax4.fill_between(x_range[x_range < 1], 0, derivative[x_range < 1], alpha=0.3, color='green')
ax4.fill_between(x_range[x_range > 1], 0, derivative[x_range > 1], alpha=0.3, color='red')

# Mark specific points to show the expansion effect
small_x = 0.1
large_x = 2.0
ax4.scatter([small_x], [1/small_x], color='green', s=100, zorder=5)
ax4.scatter([large_x], [1/large_x], color='red', s=100, zorder=5)

# Annotations explaining the gradient behavior
ax4.text(0.25, 12, r'$\frac{d}{dx}\log(0.1) = 10$' + '\n(10x sensitivity!)', fontsize=10, color='green', ha='center',
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='green', alpha=0.9))

ax4.text(2.3, 2.5, r'$\frac{d}{dx}\log(2) = 0.5$' + '\n(less sensitive)', fontsize=10, color='red', ha='center',
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='red', alpha=0.9))

# Add box explaining why this matters for ML
ax4.text(0.5, 0.7, 
         r'\textbf{Why this helps optimization:}' + '\n' +
         r'$\bullet$ Small $p$: large gradient $\rightarrow$ model updates more' + '\n' + 
         r'$\bullet$ Large $p$: small gradient $\rightarrow$ stable, no overshoot',
         transform=ax4.transAxes, fontsize=9, va='center', ha='center',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, edgecolor='orange'))

ax4.set_xlabel(r'$x$ (probability)')
ax4.set_ylabel(r'$\frac{d}{dx}\log(x)$ (gradient magnitude)')
ax4.set_title(r'Log Gradient: Stronger Signal for Small Probabilities')
ax4.set_xlim(0, 3)
ax4.set_ylim(0, 20)
ax4.legend(loc='upper right')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ml-notes/figures/log_function_why.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()

print("Generated: ml-notes/figures/log_function_why.png")
