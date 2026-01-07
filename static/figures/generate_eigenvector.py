"""
Generate eigenvector visualization showing how matrices transform vectors.
Eigenvectors only get scaled (not rotated) when multiplied by the matrix.
"""
import numpy as np
import matplotlib.pyplot as plt

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 12

# Matrix from the worked example
A = np.array([[4, 2], [1, 3]])

# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)
print(f"Matrix A:\n{A}")
print(f"Eigenvalues: {eigenvalues}")
print(f"Eigenvectors:\n{eigenvectors}")

# Create figure
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Generate unit circle points
theta = np.linspace(0, 2*np.pi, 100)
circle_x = np.cos(theta)
circle_y = np.sin(theta)
circle = np.vstack([circle_x, circle_y])

# Transform the circle
transformed = A @ circle

# Left plot: Before transformation
ax1 = axes[0]
ax1.plot(circle_x, circle_y, 'b-', lw=2, label='Unit circle')

# Plot eigenvectors (normalized)
colors = ['red', 'green']
for i in range(2):
    ev = eigenvectors[:, i]
    ev = ev / np.linalg.norm(ev)  # Normalize
    ax1.annotate('', xy=(ev[0], ev[1]), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color=colors[i], lw=3))
    ax1.text(ev[0]*1.2, ev[1]*1.2, f'$v_{i+1}$', fontsize=14, color=colors[i], fontweight='bold')

# Plot some regular vectors for comparison
regular_vectors = [np.array([1, 0]), np.array([0.5, 0.5])/np.sqrt(0.5)]
for rv in regular_vectors:
    ax1.annotate('', xy=(rv[0]*0.7, rv[1]*0.7), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5, alpha=0.7))

ax1.set_xlim(-2, 2)
ax1.set_ylim(-2, 2)
ax1.set_aspect('equal')
ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
ax1.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
ax1.set_xlabel('$x_1$')
ax1.set_ylabel('$x_2$')
ax1.set_title('Before: Unit Circle + Eigenvectors', fontsize=14)

# Right plot: After transformation
ax2 = axes[1]
ax2.plot(transformed[0], transformed[1], 'b-', lw=2, label='Transformed circle (ellipse)')

# Plot transformed eigenvectors (they should point in same direction, just scaled!)
for i in range(2):
    ev = eigenvectors[:, i]
    ev = ev / np.linalg.norm(ev)  # Normalize original
    transformed_ev = A @ ev  # Transform
    
    ax2.annotate('', xy=(transformed_ev[0], transformed_ev[1]), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color=colors[i], lw=3))
    ax2.text(transformed_ev[0]*1.1, transformed_ev[1]*1.1, 
             f'$Av_{i+1} = {eigenvalues[i]:.1f}v_{i+1}$', 
             fontsize=12, color=colors[i], fontweight='bold')

# Plot transformed regular vectors (they rotate!)
for j, rv in enumerate(regular_vectors):
    trv = A @ (rv * 0.7)
    ax2.annotate('', xy=(trv[0], trv[1]), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5, alpha=0.7))
    ax2.text(trv[0]*1.1, trv[1]*1.1, 'rotated!', fontsize=9, color='gray', alpha=0.7)

ax2.set_xlim(-6, 6)
ax2.set_ylim(-6, 6)
ax2.set_aspect('equal')
ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
ax2.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
ax2.set_xlabel('$x_1$')
ax2.set_ylabel('$x_2$')
ax2.set_title('After: $A$ Transforms Circle to Ellipse\nEigenvectors Only Scale, Don\'t Rotate!', fontsize=14)

# Add text box with eigenvalue info
textstr = f'$\\lambda_1 = {eigenvalues[0]:.1f}$ (stretch)\n$\\lambda_2 = {eigenvalues[1]:.1f}$ (stretch)'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax2.text(0.02, 0.98, textstr, transform=ax2.transAxes, fontsize=12,
        verticalalignment='top', bbox=props)

plt.tight_layout()
plt.savefig('eigenvector_transformation.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

print("\nGenerated eigenvector_transformation.png")
print(f"Matrix A transforms:")
print(f"  - Eigenvector v1 → scaled by λ1={eigenvalues[0]:.1f}")
print(f"  - Eigenvector v2 → scaled by λ2={eigenvalues[1]:.1f}")
print(f"  - Other vectors → rotated AND scaled")
