"""
Generate MCMC visualization showing Metropolis-Hastings sampling from a mixture of Gaussians.
Demonstrates how MCMC samples converge to the target distribution.
"""
import numpy as np
import matplotlib.pyplot as plt

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 12

def gaussian_pdf(x, mu, sigma):
    """Gaussian PDF using numpy only"""
    return (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def target_pdf(x):
    """Mixture of two Gaussians (unnormalized)"""
    return 0.3 * gaussian_pdf(x, -2, 0.7) + 0.7 * gaussian_pdf(x, 2, 1.0)

def metropolis_hastings(n_samples, proposal_std=1.0, seed=42):
    """Run Metropolis-Hastings algorithm"""
    np.random.seed(seed)
    samples = []
    x = 0.0  # Starting point
    
    for _ in range(n_samples):
        # Propose new point
        x_proposed = x + np.random.normal(0, proposal_std)
        
        # Acceptance ratio
        alpha = min(1, target_pdf(x_proposed) / max(target_pdf(x), 1e-10))
        
        # Accept or reject
        if np.random.random() < alpha:
            x = x_proposed
        
        samples.append(x)
    
    return np.array(samples)

# Run MCMC
n_samples = 10000
burn_in = 1000
samples = metropolis_hastings(n_samples)
samples_after_burnin = samples[burn_in:]

# Create figure
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Top-left: Trace plot (first 500 samples)
ax1 = axes[0, 0]
ax1.plot(samples[:500], 'b-', alpha=0.7, lw=0.8)
ax1.axhline(y=-2, color='r', linestyle='--', alpha=0.5, label='Mode 1 (-2)')
ax1.axhline(y=2, color='g', linestyle='--', alpha=0.5, label='Mode 2 (2)')
ax1.axvspan(0, burn_in if burn_in < 500 else 500, alpha=0.2, color='gray', label='Burn-in')
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Sample value')
ax1.set_title('MCMC Trace Plot (First 500 Samples)')
ax1.legend(loc='upper right', fontsize=9)

# Top-right: Full trace plot
ax2 = axes[0, 1]
ax2.plot(samples[::10], 'b-', alpha=0.5, lw=0.5)  # Subsample for clarity
ax2.axhline(y=-2, color='r', linestyle='--', alpha=0.5)
ax2.axhline(y=2, color='g', linestyle='--', alpha=0.5)
ax2.axvspan(0, burn_in//10, alpha=0.2, color='gray', label='Burn-in')
ax2.set_xlabel('Iteration (subsampled)')
ax2.set_ylabel('Sample value')
ax2.set_title('Full MCMC Trace (10k samples)')

# Bottom-left: Histogram vs target
ax3 = axes[1, 0]
x_range = np.linspace(-6, 6, 200)
ax3.hist(samples_after_burnin, bins=50, density=True, alpha=0.7, 
         color='steelblue', edgecolor='white', label='MCMC samples')
ax3.plot(x_range, target_pdf(x_range), 'r-', lw=2, label='Target distribution')
ax3.set_xlabel('x')
ax3.set_ylabel('Density')
ax3.set_title('MCMC Samples Match Target Distribution')
ax3.legend()

# Bottom-right: Autocorrelation
ax4 = axes[1, 1]
lags = range(0, 100)
autocorr = [np.corrcoef(samples_after_burnin[:-lag if lag > 0 else None], 
                        samples_after_burnin[lag:])[0, 1] if lag > 0 else 1.0 
            for lag in lags]
ax4.bar(lags, autocorr, color='steelblue', alpha=0.7, width=1.0)
ax4.axhline(y=0.05, color='r', linestyle='--', alpha=0.7, label='Significance threshold')
ax4.set_xlabel('Lag')
ax4.set_ylabel('Autocorrelation')
ax4.set_title('Autocorrelation of MCMC Samples')
ax4.legend()

plt.tight_layout()
plt.savefig('mcmc_sampling.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

print("Generated mcmc_sampling.png")
print(f"Samples: {n_samples}, Burn-in: {burn_in}")
print(f"Sample mean: {samples_after_burnin.mean():.3f}")
print(f"Sample std: {samples_after_burnin.std():.3f}")
