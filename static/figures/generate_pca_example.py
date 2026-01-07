"""
Generate clean PCA visualization showing 2D data projected onto first principal component.
Uses the same data from the worked example in 03-math-foundations.md
"""
import numpy as np
import matplotlib.pyplot as plt

# Set clean style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 11
plt.rcParams['font.family'] = 'sans-serif'

# Original data from the worked example
data = np.array([
    [2.5, 2.4],
    [0.5, 0.7],
    [2.2, 2.9],
    [1.9, 2.2],
    [3.1, 3.0]
])

# Center the data
mean = data.mean(axis=0)
centered = data - mean

# Compute covariance matrix
cov = np.cov(centered.T)

# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(cov)

# Sort by eigenvalue (descending)
idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

# First principal component
pc1 = eigenvectors[:, 0]

# Project data onto PC1
projections = centered @ pc1

# Create figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# ═══════════════════════════════════════════════════════════════════════════
# LEFT PANEL: Original 2D data with PC1 direction
# ═══════════════════════════════════════════════════════════════════════════
ax1 = axes[0]

# Draw PC1 direction as a line through origin (subtle background)
line_extent = 2.5
ax1.plot([-pc1[0]*line_extent, pc1[0]*line_extent], 
         [-pc1[1]*line_extent, pc1[1]*line_extent], 
         'r-', lw=2, alpha=0.4, zorder=2, label='PC1 direction')

# Draw data points (prominent)
ax1.scatter(centered[:, 0], centered[:, 1], s=120, c='#2E86AB', 
            edgecolors='white', linewidth=2, zorder=5)

# Add point numbers
for i, point in enumerate(centered):
    ax1.annotate(f'{i+1}', (point[0], point[1]), 
                 textcoords="offset points", xytext=(8, 8),
                 fontsize=10, color='#2E86AB', fontweight='bold')

# Draw PC1 arrow (bold)
arrow_scale = 1.8
ax1.annotate('', xy=(pc1[0]*arrow_scale, pc1[1]*arrow_scale), xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', color='#E63946', lw=3, 
                          mutation_scale=15))
ax1.text(pc1[0]*arrow_scale + 0.15, pc1[1]*arrow_scale + 0.1, 
         'PC1', color='#E63946', fontsize=12, fontweight='bold')

# Draw projections onto PC1 line (subtle dashed lines)
for i, point in enumerate(centered):
    proj_point = projections[i] * pc1
    ax1.plot([point[0], proj_point[0]], [point[1], proj_point[1]], 
             color='gray', linestyle=':', alpha=0.5, lw=1, zorder=3)
    # Small marker on the PC1 line
    ax1.scatter([proj_point[0]], [proj_point[1]], s=40, c='#E63946', 
                marker='|', zorder=4, linewidths=2)

# Clean up axes
ax1.set_xlabel('Feature 1 (centered)', fontsize=11)
ax1.set_ylabel('Feature 2 (centered)', fontsize=11)
ax1.set_title('Step 1: Find Direction of Max Variance', fontsize=12, fontweight='bold')
ax1.set_xlim(-2.2, 1.8)
ax1.set_ylim(-2.2, 1.8)
ax1.set_aspect('equal')
ax1.axhline(y=0, color='lightgray', linestyle='-', lw=0.5, zorder=1)
ax1.axvline(x=0, color='lightgray', linestyle='-', lw=0.5, zorder=1)

# ═══════════════════════════════════════════════════════════════════════════
# RIGHT PANEL: 1D projection result
# ═══════════════════════════════════════════════════════════════════════════
ax2 = axes[1]

# Draw the PC1 line (number line)
ax2.axhline(y=0, color='#E63946', linestyle='-', lw=3, alpha=0.7)

# Plot projected points
ax2.scatter(projections, np.zeros_like(projections), s=120, c='#2E86AB', 
            edgecolors='white', linewidth=2, zorder=5)

# Add point numbers (aligned in two rows)
# Points 4, 1, 5 above the line; Points 2, 3 below the line
offsets = [20, -25, -25, 20, 20]  # Aligned positions: above for 1,4,5; below for 2,3
for i, (proj, offset) in enumerate(zip(projections, offsets)):
    ax2.annotate(f'Point {i+1}', (proj, 0), 
                 textcoords="offset points", xytext=(0, offset),
                 fontsize=10, ha='center', fontweight='bold', color='#2E86AB')

# Add tick marks for reference
for proj in projections:
    ax2.plot([proj, proj], [-0.05, 0.05], 'k-', lw=1)

# Show variance captured
variance_pct = eigenvalues[0] / eigenvalues.sum() * 100
ax2.text(0, 0.35, f'{variance_pct:.0f}% of variance preserved', 
         ha='center', fontsize=12, fontweight='bold', color='#2E86AB',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='#E8F4F8', edgecolor='none'))

# Clean up axes
ax2.set_xlabel('PC1 (1D projection)', fontsize=11)
ax2.set_yticks([])
ax2.set_title('Step 2: Project Data onto PC1', fontsize=12, fontweight='bold')
ax2.set_xlim(-2.8, 2.0)
ax2.set_ylim(-0.6, 0.6)
ax2.spines['left'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# Overall title
fig.suptitle('PCA: Dimensionality Reduction from 2D to 1D', 
             fontsize=14, fontweight='bold', y=1.02)

plt.tight_layout()
plt.savefig('pca_projection.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

print("Generated pca_projection.png")
print(f"Eigenvalues: {eigenvalues}")
print(f"PC1 direction: {pc1}")
print(f"Variance explained by PC1: {variance_pct:.1f}%")
