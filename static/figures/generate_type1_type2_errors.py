"""
Generate Type I and Type II Errors Visualization
Shows the null and alternative hypothesis distributions with error regions
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Parameters
mu_0 = 0      # Null hypothesis mean
mu_1 = 2.5    # Alternative hypothesis mean (effect exists)
sigma = 1     # Standard deviation
alpha = 0.05  # Significance level
critical_value = stats.norm.ppf(1 - alpha, mu_0, sigma)  # One-tailed test

x = np.linspace(-4, 7, 1000)

# ============================================
# Left plot: Full visualization with both distributions
# ============================================
ax = axes[0]

# Null distribution (H0)
y_null = stats.norm.pdf(x, mu_0, sigma)
ax.plot(x, y_null, 'b-', linewidth=2.5, label='$H_0$ (No effect)', zorder=5)
ax.fill_between(x, y_null, alpha=0.15, color='blue')

# Alternative distribution (H1)
y_alt = stats.norm.pdf(x, mu_1, sigma)
ax.plot(x, y_alt, 'r-', linewidth=2.5, label='$H_1$ (Effect exists)', zorder=5)
ax.fill_between(x, y_alt, alpha=0.15, color='red')

# Critical value line
ax.axvline(critical_value, color='darkgreen', linestyle='--', linewidth=2, 
           label=f'Critical value = {critical_value:.2f}', zorder=4)

# Type I Error (α) - Area under H0 beyond critical value
x_type1 = x[x >= critical_value]
y_type1 = stats.norm.pdf(x_type1, mu_0, sigma)
ax.fill_between(x_type1, y_type1, color='blue', alpha=0.7, 
                label=f'Type I Error (α = {alpha:.2f})', zorder=3)

# Type II Error (β) - Area under H1 before critical value
x_type2 = x[x <= critical_value]
y_type2 = stats.norm.pdf(x_type2, mu_1, sigma)
beta = stats.norm.cdf(critical_value, mu_1, sigma)
ax.fill_between(x_type2, y_type2, color='red', alpha=0.7, 
                label=f'Type II Error (β = {beta:.2f})', zorder=3)

# Power region (1 - β) - Area under H1 beyond critical value
x_power = x[x >= critical_value]
y_power = stats.norm.pdf(x_power, mu_1, sigma)
power = 1 - beta
ax.fill_between(x_power, y_power, color='green', alpha=0.4, 
                label=f'Power (1-β = {power:.2f})', zorder=2)

# Labels and annotations
ax.annotate('Reject $H_0$', xy=(critical_value + 1.5, 0.35), fontsize=12, fontweight='bold',
            ha='center', color='darkgreen')
ax.annotate('Fail to reject $H_0$', xy=(critical_value - 1.5, 0.35), fontsize=12, fontweight='bold',
            ha='center', color='gray')

# Add arrow showing decision boundary
ax.annotate('', xy=(critical_value + 0.3, 0.42), xytext=(critical_value - 0.3, 0.42),
            arrowprops=dict(arrowstyle='<->', color='darkgreen', lw=2))

ax.set_xlabel('Test Statistic Value', fontsize=12)
ax.set_ylabel('Probability Density', fontsize=12)
ax.set_title('Type I and Type II Errors in Hypothesis Testing', fontsize=14, fontweight='bold')
ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
ax.set_xlim(-4, 7)
ax.set_ylim(0, 0.5)

# ============================================
# Right plot: Confusion matrix style interpretation
# ============================================
ax = axes[1]
ax.axis('off')

# Create a table/matrix visualization
cell_text = [
    ['', 'Reality: H0 True\n(No effect)', 'Reality: H0 False\n(Effect exists)'],
    ['Decision:\nReject H0', 'TYPE I ERROR\n(False Positive)\n\na = P(reject H0 | H0 true)', 'CORRECT\n(True Positive)\n\nPower = 1 - b'],
    ['Decision:\nFail to Reject H0', 'CORRECT\n(True Negative)\n\n1 - a', 'TYPE II ERROR\n(False Negative)\n\nb = P(fail to reject | H0 false)']
]

# Draw the matrix
colors = [
    ['white', 'lightblue', 'lightcoral'],
    ['lightyellow', '#ffcccc', '#ccffcc'],  # FP is red-ish, TP is green
    ['lightyellow', '#ccffcc', '#ffcccc']   # TN is green, FN is red-ish
]

table = ax.table(cellText=cell_text, 
                 cellColours=colors,
                 cellLoc='center',
                 loc='center',
                 bbox=[0, 0, 1, 1])

table.auto_set_font_size(False)
table.set_fontsize(10)

# Style the cells
for (row, col), cell in table.get_celld().items():
    cell.set_height(0.25)
    if row == 0 or col == 0:
        cell.set_text_props(fontweight='bold')
    if row == 0:
        cell.set_height(0.15)
    if col == 0:
        cell.set_width(0.2)

ax.set_title('Hypothesis Testing Confusion Matrix', fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.subplots_adjust(bottom=0.18)

# Add memory tricks at the bottom (after tight_layout)
fig.text(0.5, 0.01, 
         'Memory Trick: Type I = "False Alarm" (rejected true $H_0$)  •  Type II = "Missed Detection" (failed to reject false $H_0$)',
         ha='center', fontsize=10, style='italic', 
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
plt.savefig('ml-notes/figures/type1_type2_errors.png', dpi=150, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.close()

print("Generated: type1_type2_errors.png")
