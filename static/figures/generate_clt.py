"""
Generate Central Limit Theorem visualization.
Shows how sample means converge to normal distribution regardless of original distribution.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Set style
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['figure.facecolor'] = 'white'

np.random.seed(42)

def sample_means(dist_func, n_samples, sample_size, **kwargs):
    """Generate n_samples sample means, each from sample_size observations."""
    means = []
    for _ in range(n_samples):
        sample = dist_func(size=sample_size, **kwargs)
        means.append(np.mean(sample))
    return np.array(means)

# Create figure with 3 rows x 4 columns
fig, axes = plt.subplots(3, 4, figsize=(14, 10))

# Number of sample means to generate
n_samples = 10000

# Three different original distributions
distributions = [
    ("Uniform [0,1]", lambda size: np.random.uniform(0, 1, size), 0.5, np.sqrt(1/12)),
    ("Exponential (λ=1)", lambda size: np.random.exponential(1, size), 1.0, 1.0),
    ("Bimodal", lambda size: np.where(np.random.random(size) < 0.5, 
                                       np.random.normal(-2, 0.5, size),
                                       np.random.normal(2, 0.5, size)), 0.0, np.sqrt(4.25)),
]

# Sample sizes to show
sample_sizes = [1, 5, 30]

for row, (dist_name, dist_func, mu, sigma) in enumerate(distributions):
    # Column 0: Original distribution
    ax = axes[row, 0]
    original_samples = dist_func(size=10000)
    ax.hist(original_samples, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='white')
    ax.set_title(f"Original: {dist_name}")
    ax.set_ylabel(f"{dist_name}" if row == 1 else "")
    if row == 0:
        ax.set_xlabel("")
    
    # Add mean line
    ax.axvline(mu, color='red', linestyle='--', linewidth=2, label=f'μ = {mu:.1f}')
    if row == 0:
        ax.legend(loc='upper right', fontsize=8)
    
    # Columns 1-3: Sample means for different n
    for col, n in enumerate(sample_sizes):
        ax = axes[row, col + 1]
        
        # Generate sample means
        means = sample_means(dist_func, n_samples, n)
        
        # Plot histogram
        ax.hist(means, bins=50, density=True, alpha=0.7, color='coral', edgecolor='white')
        
        # Overlay theoretical normal (from CLT)
        x = np.linspace(means.min(), means.max(), 100)
        std_of_mean = sigma / np.sqrt(n)
        normal_pdf = stats.norm.pdf(x, mu, std_of_mean)
        ax.plot(x, normal_pdf, 'k-', linewidth=2, label='Normal (CLT)')
        
        # Title
        if row == 0:
            ax.set_title(f"Sample Mean (n={n})")
        
        # Add annotation for variance
        if row == 2:
            ax.set_xlabel(f"σ/√n = {std_of_mean:.2f}")
        
        # Legend on first row only
        if row == 0 and col == 2:
            ax.legend(loc='upper right', fontsize=8)

# Main title
fig.suptitle("Central Limit Theorem: Sample Means → Normal Distribution\n" + 
             "As sample size n increases, the distribution of sample means becomes normal,\n" +
             "regardless of the original distribution!", fontsize=13, fontweight='bold', y=1.02)

plt.tight_layout()
plt.savefig('clt_visualization.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

print("Generated: clt_visualization.png")

# Also create a simpler single-row figure focusing on one distribution
fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))

# Use dice rolling as the example (discrete uniform 1-6)
def roll_dice(size):
    return np.random.randint(1, 7, size)

mu_dice = 3.5
sigma_dice = np.sqrt(35/12)  # Variance of uniform discrete 1-6

# Original distribution
ax = axes[0]
dice_samples = roll_dice(10000)
counts = [np.sum(dice_samples == i) for i in range(1, 7)]
ax.bar(range(1, 7), np.array(counts)/10000, color='steelblue', edgecolor='white', alpha=0.8)
ax.set_title("Original: Single Die Roll\n(Discrete Uniform 1-6)")
ax.set_xlabel("Die Face")
ax.set_ylabel("Probability")
ax.set_xticks(range(1, 7))
ax.axhline(1/6, color='red', linestyle='--', linewidth=2, alpha=0.7)

# Sample means for n = 2, 10, 30
for idx, n in enumerate([2, 10, 30]):
    ax = axes[idx + 1]
    means = sample_means(roll_dice, n_samples, n)
    
    ax.hist(means, bins=40, density=True, alpha=0.7, color='coral', edgecolor='white')
    
    # Overlay CLT prediction
    x = np.linspace(means.min(), means.max(), 100)
    std_of_mean = sigma_dice / np.sqrt(n)
    normal_pdf = stats.norm.pdf(x, mu_dice, std_of_mean)
    ax.plot(x, normal_pdf, 'k-', linewidth=2, label='CLT Normal')
    
    ax.set_title(f"Mean of n={n} Dice Rolls")
    ax.set_xlabel(f"Sample Mean\n(σ/√n = {std_of_mean:.2f})")
    if idx == 2:
        ax.legend(loc='upper right', fontsize=9)

fig.suptitle("Central Limit Theorem with Dice: Sample Means Become Normal!", 
             fontsize=12, fontweight='bold', y=1.05)
plt.tight_layout()
plt.savefig('clt_dice_example.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

print("Generated: clt_dice_example.png")
