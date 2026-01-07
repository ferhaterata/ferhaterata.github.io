"""
Generate Z-Test vs T-Test and P-Value Visualization
Shows t-distribution convergence, p-value interpretation, and one/two-tailed tests
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

x = np.linspace(-4.5, 4.5, 500)

# ============================================
# Left panel: T-distribution vs Normal
# ============================================
ax = axes[0]

# Standard normal
y_normal = stats.norm.pdf(x)
ax.plot(x, y_normal, 'b-', linewidth=2.5, label='Normal (z)', zorder=5)

# T-distributions with different df
df_values = [3, 10, 30]
colors = ['red', 'orange', 'green']
for df, color in zip(df_values, colors):
    y_t = stats.t.pdf(x, df)
    ax.plot(x, y_t, color=color, linewidth=2, linestyle='--', 
            label=f't (df={df})', alpha=0.8)

# Highlight the tails
ax.fill_between(x[x < -2.5], stats.norm.pdf(x[x < -2.5]), alpha=0.2, color='blue')
ax.fill_between(x[x > 2.5], stats.norm.pdf(x[x > 2.5]), alpha=0.2, color='blue')

# Add annotation about tails
ax.annotate('Heavier tails\nfor small df', xy=(-3.2, 0.02), fontsize=9,
            ha='center', color='red')
ax.annotate('Heavier tails\nfor small df', xy=(3.2, 0.02), fontsize=9,
            ha='center', color='red')

ax.set_xlabel('Test Statistic Value')
ax.set_ylabel('Probability Density')
ax.set_title('T-Distribution vs Normal\n(t converges to Normal as df increases)', fontweight='bold')
ax.legend(loc='upper right', fontsize=9)
ax.set_xlim(-4.5, 4.5)
ax.set_ylim(0, 0.45)

# Add key insight box
textstr = 'Use t-test when:\n- Population variance unknown\n- Sample size is small'
props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.8)
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=8,
        verticalalignment='top', bbox=props)

# ============================================
# Middle panel: P-value visualization
# ============================================
ax = axes[1]

# Distribution under null hypothesis
y = stats.norm.pdf(x)
ax.plot(x, y, 'b-', linewidth=2.5, label='Distribution under H0')
ax.fill_between(x, y, alpha=0.15, color='blue')

# Observed test statistic
observed = 2.1
ax.axvline(observed, color='red', linewidth=2.5, linestyle='-', label=f'Observed statistic = {observed}')

# Shade p-value region (one-tailed, right)
x_pvalue = x[x >= observed]
y_pvalue = stats.norm.pdf(x_pvalue)
ax.fill_between(x_pvalue, y_pvalue, color='red', alpha=0.6, 
                label=f'P-value = {1 - stats.norm.cdf(observed):.4f}')

# Add annotation
ax.annotate('P-value = P(data this extreme\nor more | H0 true)', 
            xy=(observed + 0.3, 0.08), fontsize=9,
            xytext=(observed + 0.8, 0.20),
            arrowprops=dict(arrowstyle='->', color='red'),
            color='red')

ax.set_xlabel('Test Statistic Value')
ax.set_ylabel('Probability Density')
ax.set_title('P-Value: Probability of Extreme Data\n(One-tailed test)', fontweight='bold')
ax.legend(loc='upper left', fontsize=9)
ax.set_xlim(-4.5, 4.5)
ax.set_ylim(0, 0.45)

# Add interpretation
textstr = 'Small p-value:\nData unlikely under H0\n=> Reject H0'
props = dict(boxstyle='round', facecolor='lightcoral', alpha=0.5)
ax.text(0.98, 0.98, textstr, transform=ax.transAxes, fontsize=8,
        verticalalignment='top', ha='right', bbox=props)

# ============================================
# Right panel: One-tailed vs Two-tailed
# ============================================
ax = axes[2]

# Distribution
y = stats.norm.pdf(x)
ax.plot(x, y, 'b-', linewidth=2.5)
ax.fill_between(x, y, alpha=0.1, color='blue')

alpha = 0.05

# Two-tailed critical values
z_crit_two = stats.norm.ppf(1 - alpha/2)  # 1.96

# One-tailed critical value
z_crit_one = stats.norm.ppf(1 - alpha)  # 1.645

# Shade two-tailed rejection regions
x_left = x[x <= -z_crit_two]
x_right = x[x >= z_crit_two]
ax.fill_between(x_left, stats.norm.pdf(x_left), color='purple', alpha=0.5, label='Two-tailed (alpha/2 each)')
ax.fill_between(x_right, stats.norm.pdf(x_right), color='purple', alpha=0.5)

# Mark critical values
ax.axvline(-z_crit_two, color='purple', linewidth=2, linestyle='--')
ax.axvline(z_crit_two, color='purple', linewidth=2, linestyle='--')
ax.axvline(z_crit_one, color='green', linewidth=2, linestyle=':', label=f'One-tailed (alpha={alpha})')

# Add annotations
ax.annotate(f'-{z_crit_two:.2f}', xy=(-z_crit_two, 0.01), fontsize=9, ha='center', color='purple')
ax.annotate(f'+{z_crit_two:.2f}', xy=(z_crit_two, 0.01), fontsize=9, ha='center', color='purple')
ax.annotate(f'+{z_crit_one:.2f}', xy=(z_crit_one, 0.02), fontsize=9, ha='left', color='green')

ax.annotate(f'alpha/2 = {alpha/2}', xy=(-3.5, 0.06), fontsize=9, color='purple')
ax.annotate(f'alpha/2 = {alpha/2}', xy=(2.5, 0.06), fontsize=9, color='purple')

ax.set_xlabel('Test Statistic Value')
ax.set_ylabel('Probability Density')
ax.set_title('One-Tailed vs Two-Tailed Tests\n(Significance level alpha = 0.05)', fontweight='bold')
ax.legend(loc='upper left', fontsize=9)
ax.set_xlim(-4.5, 4.5)
ax.set_ylim(0, 0.45)

# Add comparison box
textstr = 'Two-tailed: Tests H1: mu != mu0\nOne-tailed: Tests H1: mu > mu0\n                   or H1: mu < mu0'
props = dict(boxstyle='round', facecolor='lightgreen', alpha=0.5)
ax.text(0.98, 0.98, textstr, transform=ax.transAxes, fontsize=8,
        verticalalignment='top', ha='right', bbox=props)

plt.tight_layout()
plt.savefig('ml-notes/figures/z_t_test_pvalue.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()

print("Generated: z_t_test_pvalue.png")
