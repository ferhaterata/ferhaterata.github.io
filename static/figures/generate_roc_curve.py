"""
Generate ROC curve visualization for ML fundamentals chapter.
"""

import numpy as np
import matplotlib.pyplot as plt

# Set style
plt.style.use('default')
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['figure.facecolor'] = 'white'


def compute_roc_curve(y_true, y_scores, n_thresholds=100):
    """Compute ROC curve without sklearn."""
    thresholds = np.linspace(0, 1, n_thresholds)
    tpr_list = []
    fpr_list = []
    
    for thresh in thresholds:
        y_pred = (y_scores >= thresh).astype(int)
        
        # True positives, false positives, etc.
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        tn = np.sum((y_pred == 0) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        
        # TPR and FPR
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        tpr_list.append(tpr)
        fpr_list.append(fpr)
    
    return np.array(fpr_list), np.array(tpr_list), thresholds


def compute_auc(fpr, tpr):
    """Compute AUC using trapezoidal rule."""
    # Sort by FPR
    sorted_indices = np.argsort(fpr)
    fpr_sorted = fpr[sorted_indices]
    tpr_sorted = tpr[sorted_indices]
    return np.trapz(tpr_sorted, fpr_sorted)


# Generate synthetic data for different classifiers
np.random.seed(42)
n_samples = 1000

# True labels (imbalanced: 30% positive)
y_true = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])

# Simulate different classifier outputs
# Good classifier: higher scores for positive class
y_scores_good = np.where(y_true == 1, 
                         np.random.beta(5, 2, n_samples),
                         np.random.beta(2, 5, n_samples))

# Medium classifier: some separation
y_scores_medium = np.where(y_true == 1,
                           np.random.beta(3, 2, n_samples),
                           np.random.beta(2, 3, n_samples))

# Poor classifier: almost random
y_scores_poor = np.random.uniform(0, 1, n_samples)

# Compute ROC curves
fpr_good, tpr_good, thresh_good = compute_roc_curve(y_true, y_scores_good)
fpr_medium, tpr_medium, _ = compute_roc_curve(y_true, y_scores_medium)
fpr_poor, tpr_poor, _ = compute_roc_curve(y_true, y_scores_poor)

# Compute AUC
auc_good = compute_auc(fpr_good, tpr_good)
auc_medium = compute_auc(fpr_medium, tpr_medium)
auc_poor = compute_auc(fpr_poor, tpr_poor)

# Create figure
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left plot: ROC Curves Comparison
ax1 = axes[0]
ax1.plot(fpr_good, tpr_good, 'b-', linewidth=2.5, label=f'Good Classifier (AUC = {auc_good:.2f})')
ax1.plot(fpr_medium, tpr_medium, 'g-', linewidth=2.5, label=f'Medium Classifier (AUC = {auc_medium:.2f})')
ax1.plot(fpr_poor, tpr_poor, 'r-', linewidth=2.5, label=f'Random Classifier (AUC = {auc_poor:.2f})')
ax1.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random Baseline')

# Shade the area under the good classifier's curve
ax1.fill_between(fpr_good, tpr_good, alpha=0.2, color='blue')

ax1.set_xlabel('False Positive Rate (FPR)\n"Of all negatives, how many did I wrongly flag?"')
ax1.set_ylabel('True Positive Rate (TPR) = Recall\n"Of all positives, how many did I catch?"')
ax1.set_title('ROC Curves: Comparing Classifiers')
ax1.legend(loc='lower right', fontsize=10)
ax1.set_xlim([0, 1])
ax1.set_ylim([0, 1])
ax1.grid(True, alpha=0.3)
ax1.set_aspect('equal')

# Add annotation for the "elbow" region
ax1.annotate('Best operating\npoint region', 
            xy=(0.1, 0.85), xytext=(0.3, 0.6),
            fontsize=10, ha='center',
            arrowprops=dict(arrowstyle='->', color='blue', lw=1.5))

# Right plot: Threshold effect demonstration
ax2 = axes[1]

# Show how different thresholds affect the point on the ROC curve
thresholds = [0.3, 0.5, 0.7]
colors = ['green', 'orange', 'red']
markers = ['o', 's', '^']

ax2.plot(fpr_good, tpr_good, 'b-', linewidth=2, alpha=0.7)
ax2.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.5)

# Mark specific threshold points
for thresh, color, marker in zip(thresholds, colors, markers):
    # Find closest threshold
    thresh_idx = np.argmin(np.abs(thresh_good - thresh))
    ax2.scatter(fpr_good[thresh_idx], tpr_good[thresh_idx], 
               c=color, s=150, marker=marker, zorder=5,
               label=f'Threshold = {thresh}', edgecolors='black', linewidth=1.5)

ax2.set_xlabel('False Positive Rate (FPR)')
ax2.set_ylabel('True Positive Rate (TPR)')
ax2.set_title('ROC Curve: Effect of Classification Threshold')
ax2.legend(loc='lower right', fontsize=10)
ax2.set_xlim([0, 1])
ax2.set_ylim([0, 1])
ax2.grid(True, alpha=0.3)
ax2.set_aspect('equal')

# Add text annotations
ax2.text(0.6, 0.3, 'Higher threshold:\n• Fewer false positives\n• Fewer true positives\n(More conservative)', 
        fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
ax2.text(0.05, 0.7, 'Lower threshold:\n• More true positives\n• More false positives\n(More liberal)', 
        fontsize=9, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

plt.tight_layout()
plt.savefig('roc_curve.png', dpi=150, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.close()

print("Generated roc_curve.png")
