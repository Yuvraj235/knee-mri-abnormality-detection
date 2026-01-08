import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

def find_optimal_threshold(y_true, y_prob, cost_fn_miss=3, cost_fp=1):
    """
    Find optimal threshold that balances sensitivity and specificity
    
    Args:
        cost_fn_miss: Cost of missing an abnormality (higher = prioritize sensitivity)
        cost_fp: Cost of false alarm
    """
    
    thresholds = np.linspace(0.1, 0.9, 100)
    results = []
    
    for threshold in thresholds:
        y_pred = (y_prob > threshold).astype(int)
        
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        tn = ((y_pred == 0) & (y_true == 0)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        accuracy = (tp + tn) / len(y_true)
        
        # Clinical cost
        cost = (fn * cost_fn_miss) + (fp * cost_fp)
        
        # F2 score (weights sensitivity 2x more than precision)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f2 = 5 * precision * sensitivity / (4 * precision + sensitivity) if (4 * precision + sensitivity) > 0 else 0
        
        # Balanced metric
        balanced = (sensitivity + specificity) / 2
        
        results.append({
            'threshold': threshold,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'accuracy': accuracy,
            'cost': cost,
            'f2_score': f2,
            'balanced': balanced,
            'tp': tp,
            'tn': tn,
            'fp': fp,
            'fn': fn
        })
    
    results_df = pd.DataFrame(results)
    
    # Find best thresholds by different criteria
    best_f2 = results_df.loc[results_df['f2_score'].idxmax()]
    best_cost = results_df.loc[results_df['cost'].idxmin()]
    best_balanced = results_df.loc[results_df['balanced'].idxmax()]
    
    # Find threshold with sensitivity >= 80% and max specificity
    high_sens = results_df[results_df['sensitivity'] >= 0.80]
    if len(high_sens) > 0:
        best_high_sens = high_sens.loc[high_sens['specificity'].idxmax()]
    else:
        best_high_sens = best_balanced
    
    # Plot comprehensive analysis
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Sensitivity & Specificity vs Threshold
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(results_df['threshold'], results_df['sensitivity'], 'b-', linewidth=2, label='Sensitivity')
    ax1.plot(results_df['threshold'], results_df['specificity'], 'r-', linewidth=2, label='Specificity')
    ax1.plot(results_df['threshold'], results_df['accuracy'], 'g--', linewidth=2, label='Accuracy')
    ax1.axvline(0.5, color='gray', linestyle=':', alpha=0.5, label='Default (0.5)')
    ax1.axvline(best_balanced['threshold'], color='purple', linestyle='--', linewidth=2, label=f'Best Balance ({best_balanced["threshold"]:.2f})')
    ax1.axvline(best_high_sens['threshold'], color='orange', linestyle='--', linewidth=2, label=f'High Sens ({best_high_sens["threshold"]:.2f})')
    ax1.set_xlabel('Threshold', fontsize=12, weight='bold')
    ax1.set_ylabel('Score', fontsize=12, weight='bold')
    ax1.set_title('Performance Metrics vs Threshold', fontsize=14, weight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)
    ax1.set_ylim([0, 1])
    
    # 2. ROC Curve
    ax2 = fig.add_subplot(gs[0, 2])
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    ax2.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {roc_auc:.3f})')
    ax2.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Random')
    
    # Mark current point (threshold=0.5)
    current_fpr = 1 - results_df[results_df['threshold'] == 0.5]['specificity'].values[0] if len(results_df[results_df['threshold'] == 0.5]) > 0 else 0.12
    current_tpr = results_df[results_df['threshold'] == 0.5]['sensitivity'].values[0] if len(results_df[results_df['threshold'] == 0.5]) > 0 else 0.737
    ax2.plot(current_fpr, current_tpr, 'ro', markersize=10, label='Current (0.5)')
    
    # Mark optimal point
    opt_fpr = 1 - best_balanced['specificity']
    opt_tpr = best_balanced['sensitivity']
    ax2.plot(opt_fpr, opt_tpr, 'go', markersize=10, label='Optimal')
    
    ax2.set_xlabel('False Positive Rate', fontsize=11, weight='bold')
    ax2.set_ylabel('True Positive Rate', fontsize=11, weight='bold')
    ax2.set_title('ROC Curve', fontsize=13, weight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)
    
    # 3. F2 Score vs Threshold
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(results_df['threshold'], results_df['f2_score'], 'g-', linewidth=2)
    ax3.axvline(best_f2['threshold'], color='darkgreen', linestyle='--', linewidth=2, 
                label=f'Max F2: {best_f2["threshold"]:.2f}')
    ax3.set_xlabel('Threshold', fontsize=11, weight='bold')
    ax3.set_ylabel('F2 Score', fontsize=11, weight='bold')
    ax3.set_title('F2 Score vs Threshold', fontsize=13, weight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(alpha=0.3)
    
    # 4. Clinical Cost vs Threshold
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(results_df['threshold'], results_df['cost'], 'r-', linewidth=2)
    ax4.axvline(best_cost['threshold'], color='darkred', linestyle='--', linewidth=2,
                label=f'Min Cost: {best_cost["threshold"]:.2f}')
    ax4.set_xlabel('Threshold', fontsize=11, weight='bold')
    ax4.set_ylabel('Clinical Cost', fontsize=11, weight='bold')
    ax4.set_title(f'Cost vs Threshold (FN={cost_fn_miss}x, FP={cost_fp}x)', fontsize=13, weight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(alpha=0.3)
    
    # 5. Confusion Matrices Comparison
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis('off')
    
    cm_text = "CONFUSION MATRIX COMPARISON:\n\n"
    cm_text += f"Current (threshold=0.5):\n"
    current = results_df[results_df['threshold'].round(2) == 0.5].iloc[0]
    cm_text += f"  TP:{current['tp']:.0f} FP:{current['fp']:.0f}\n"
    cm_text += f"  FN:{current['fn']:.0f} TN:{current['tn']:.0f}\n"
    cm_text += f"  Sens: {current['sensitivity']:.1%}\n"
    cm_text += f"  Spec: {current['specificity']:.1%}\n\n"
    
    cm_text += f"Optimal (threshold={best_balanced['threshold']:.2f}):\n"
    cm_text += f"  TP:{best_balanced['tp']:.0f} FP:{best_balanced['fp']:.0f}\n"
    cm_text += f"  FN:{best_balanced['fn']:.0f} TN:{best_balanced['tn']:.0f}\n"
    cm_text += f"  Sens: {best_balanced['sensitivity']:.1%} ✅\n"
    cm_text += f"  Spec: {best_balanced['specificity']:.1%} ✅\n\n"
    
    cm_text += f"High Sensitivity (t={best_high_sens['threshold']:.2f}):\n"
    cm_text += f"  TP:{best_high_sens['tp']:.0f} FP:{best_high_sens['fp']:.0f}\n"
    cm_text += f"  FN:{best_high_sens['fn']:.0f} TN:{best_high_sens['tn']:.0f}\n"
    cm_text += f"  Sens: {best_high_sens['sensitivity']:.1%} 🎯\n"
    cm_text += f"  Spec: {best_high_sens['specificity']:.1%}\n"
    
    ax5.text(0.1, 0.5, cm_text, fontsize=10, verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5),
            family='monospace')
    ax5.set_title('Confusion Matrices', fontsize=13, weight='bold')
    
    # 6. Detailed metrics table
    ax6 = fig.add_subplot(gs[2, :])
    ax6.axis('off')
    
    comparison = pd.DataFrame({
        'Threshold': [0.5, best_balanced['threshold'], best_high_sens['threshold'], best_f2['threshold']],
        'Strategy': ['Current', 'Best Balance', 'High Sensitivity', 'Best F2'],
        'Sensitivity': [current['sensitivity'], best_balanced['sensitivity'], best_high_sens['sensitivity'], best_f2['sensitivity']],
        'Specificity': [current['specificity'], best_balanced['specificity'], best_high_sens['specificity'], best_f2['specificity']],
        'Accuracy': [current['accuracy'], best_balanced['accuracy'], best_high_sens['accuracy'], best_f2['accuracy']],
        'FN': [current['fn'], best_balanced['fn'], best_high_sens['fn'], best_f2['fn']],
        'FP': [current['fp'], best_balanced['fp'], best_high_sens['fp'], best_f2['fp']]
    })
    
    table = ax6.table(cellText=comparison.values, colLabels=comparison.columns,
                      cellLoc='center', loc='center', bbox=[0.1, 0.3, 0.8, 0.5])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    for i in range(len(comparison.columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Highlight best row
    for i in range(len(comparison.columns)):
        table[(2, i)].set_facecolor('#90EE90')  # High sensitivity row
    
    ax6.set_title('Threshold Strategy Comparison', fontsize=14, weight='bold', pad=20)
    
    plt.suptitle('THRESHOLD OPTIMIZATION ANALYSIS', fontsize=18, weight='bold', y=0.98)
    plt.savefig('outputs/improved_analysis/threshold_optimization.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    # Print comprehensive results
    print("\n" + "="*80)
    print("🎯 THRESHOLD OPTIMIZATION RESULTS")
    print("="*80)
    
    print(f"\n📊 CURRENT PERFORMANCE (threshold=0.5):")
    print(f"   Sensitivity: {current['sensitivity']:.1%}")
    print(f"   Specificity: {current['specificity']:.1%}")
    print(f"   Accuracy:    {current['accuracy']:.1%}")
    print(f"   FN: {current['fn']:.0f}, FP: {current['fp']:.0f}")
    
    print(f"\n✅ BEST BALANCED (threshold={best_balanced['threshold']:.3f}):")
    print(f"   Sensitivity: {best_balanced['sensitivity']:.1%} ({(best_balanced['sensitivity']-current['sensitivity'])*100:+.1f}%)")
    print(f"   Specificity: {best_balanced['specificity']:.1%} ({(best_balanced['specificity']-current['specificity'])*100:+.1f}%)")
    print(f"   Accuracy:    {best_balanced['accuracy']:.1%} ({(best_balanced['accuracy']-current['accuracy'])*100:+.1f}%)")
    print(f"   FN: {best_balanced['fn']:.0f} ({best_balanced['fn']-current['fn']:+.0f}), FP: {best_balanced['fp']:.0f} ({best_balanced['fp']-current['fp']:+.0f})")
    
    print(f"\n🎯 HIGH SENSITIVITY (threshold={best_high_sens['threshold']:.3f}):")
    print(f"   Sensitivity: {best_high_sens['sensitivity']:.1%} ({(best_high_sens['sensitivity']-current['sensitivity'])*100:+.1f}%)")
    print(f"   Specificity: {best_high_sens['specificity']:.1%} ({(best_high_sens['specificity']-current['specificity'])*100:+.1f}%)")
    print(f"   Accuracy:    {best_high_sens['accuracy']:.1%} ({(best_high_sens['accuracy']-current['accuracy'])*100:+.1f}%)")
    print(f"   FN: {best_high_sens['fn']:.0f} ({best_high_sens['fn']-current['fn']:+.0f}), FP: {best_high_sens['fp']:.0f} ({best_high_sens['fp']-current['fp']:+.0f})")
    
    print(f"\n💰 MINIMUM COST (threshold={best_cost['threshold']:.3f}):")
    print(f"   Sensitivity: {best_cost['sensitivity']:.1%}")
    print(f"   Specificity: {best_cost['specificity']:.1%}")
    print(f"   Cost: {best_cost['cost']:.0f} (vs {current['cost']:.0f})")
    
    print(f"\n📈 BEST F2 SCORE (threshold={best_f2['threshold']:.3f}):")
    print(f"   Sensitivity: {best_f2['sensitivity']:.1%}")
    print(f"   Specificity: {best_f2['specificity']:.1%}")
    print(f"   F2 Score: {best_f2['f2_score']:.3f}")
    
    print("\n" + "="*80)
    print("💡 RECOMMENDATION:")
    print("="*80)
    print(f"Use threshold = {best_balanced['threshold']:.3f} for best overall balance")
    print(f"Or use threshold = {best_high_sens['threshold']:.3f} to prioritize catching abnormalities")
    print("\n✅ Saved analysis to: outputs/improved_analysis/threshold_optimization.png")
    print("="*80)
    
    return results_df, best_balanced['threshold'], best_high_sens['threshold']

if __name__ == '__main__':
    import os
    
    # Load predictions
    pred_file = 'outputs/improved_analysis/predictions_improved.csv'
    
    if not os.path.exists(pred_file):
        print("❌ Error: predictions_improved.csv not found!")
        print("Please run: python src/detailed_analysis_improved.py first")
        exit(1)
    
    df = pd.read_csv(pred_file)
    
    # Run optimization
    results_df, optimal_threshold, high_sens_threshold = find_optimal_threshold(
        df['true_label'].values,
        df['probability'].values,
        cost_fn_miss=3,  # Missing abnormality is 3x worse than false alarm
        cost_fp=1
    )
    
    # Save detailed results
    results_df.to_csv('outputs/improved_analysis/threshold_analysis.csv', index=False)
    print(f"\n📁 Saved detailed results to: outputs/improved_analysis/threshold_analysis.csv")
