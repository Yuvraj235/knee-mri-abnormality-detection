import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def main():
    print("="*70)
    print("📊 COMPLETE MODEL COMPARISON")
    print("="*70)
    
    # Load all predictions
    try:
        improved_df = pd.read_csv('outputs/improved_analysis/predictions_improved.csv')
        multiplane_df = pd.read_csv('outputs/multiplane_analysis/predictions_multiplane.csv')
    except:
        print("❌ Error: Please run analysis scripts first!")
        return
    
    # Calculate metrics
    def calc_metrics(df):
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(df['true_label'], df['prediction'])
        tn, fp, fn, tp = cm.ravel()
        return {
            'sensitivity': tp/(tp+fn) if (tp+fn) > 0 else 0,
            'specificity': tn/(tn+fp) if (tn+fp) > 0 else 0,
            'accuracy': (tp+tn)/len(df),
            'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
        }
    
    old_metrics = {'sensitivity': 0.958, 'specificity': 0.400, 'accuracy': 0.842}
    improved_metrics = calc_metrics(improved_df)
    multiplane_metrics = calc_metrics(multiplane_df)
    
    # Create comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    models = ['Old\nBaseline', 'Single-Plane\nImproved', 'Multi-Plane\nFusion']
    
    # Sensitivity
    sens_vals = [old_metrics['sensitivity'], improved_metrics['sensitivity'], multiplane_metrics['sensitivity']]
    axes[0, 0].bar(models, sens_vals, color=['lightcoral', 'lightblue', 'lightgreen'])
    axes[0, 0].set_title('Sensitivity Comparison', fontsize=14, weight='bold')
    axes[0, 0].set_ylim([0, 1])
    axes[0, 0].axhline(y=0.9, color='orange', linestyle='--', alpha=0.5)
    for i, v in enumerate(sens_vals):
        axes[0, 0].text(i, v, f'{v:.1%}', ha='center', va='bottom', fontsize=11, weight='bold')
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # Specificity
    spec_vals = [old_metrics['specificity'], improved_metrics['specificity'], multiplane_metrics['specificity']]
    axes[0, 1].bar(models, spec_vals, color=['red', 'green', 'yellow'])
    axes[0, 1].set_title('Specificity Comparison', fontsize=14, weight='bold')
    axes[0, 1].set_ylim([0, 1])
    axes[0, 1].axhline(y=0.65, color='orange', linestyle='--', alpha=0.5)
    for i, v in enumerate(spec_vals):
        axes[0, 1].text(i, v, f'{v:.1%}', ha='center', va='bottom', fontsize=11, weight='bold')
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Accuracy
    acc_vals = [old_metrics['accuracy'], improved_metrics['accuracy'], multiplane_metrics['accuracy']]
    axes[1, 0].bar(models, acc_vals, color=['lightyellow', 'lightcyan', 'lightpink'])
    axes[1, 0].set_title('Accuracy Comparison', fontsize=14, weight='bold')
    axes[1, 0].set_ylim([0, 1])
    for i, v in enumerate(acc_vals):
        axes[1, 0].text(i, v, f'{v:.1%}', ha='center', va='bottom', fontsize=11, weight='bold')
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # Summary table
    axes[1, 1].axis('off')
    summary_text = "MODEL COMPARISON SUMMARY\n\n"
    summary_text += "OLD BASELINE:\n"
    summary_text += f"  Sens: {old_metrics['sensitivity']:.1%}\n"
    summary_text += f"  Spec: {old_metrics['specificity']:.1%} ❌\n"
    summary_text += f"  Acc:  {old_metrics['accuracy']:.1%}\n\n"
    
    summary_text += "SINGLE-PLANE IMPROVED:\n"
    summary_text += f"  Sens: {improved_metrics['sensitivity']:.1%}\n"
    summary_text += f"  Spec: {improved_metrics['specificity']:.1%} ✅\n"
    summary_text += f"  Acc:  {improved_metrics['accuracy']:.1%}\n\n"
    
    summary_text += "MULTI-PLANE FUSION:\n"
    summary_text += f"  Sens: {multiplane_metrics['sensitivity']:.1%} 🎯\n"
    summary_text += f"  Spec: {multiplane_metrics['specificity']:.1%}\n"
    summary_text += f"  Acc:  {multiplane_metrics['accuracy']:.1%}\n\n"
    
    summary_text += "BEST APPROACH:\n"
    summary_text += "Multi-Plane for detection\n"
    summary_text += "Single-Plane for precision"
    
    axes[1, 1].text(0.1, 0.5, summary_text, fontsize=11, verticalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5),
                    family='monospace')
    
    plt.suptitle('COMPLETE MODEL COMPARISON', fontsize=18, weight='bold')
    plt.tight_layout()
    plt.savefig('outputs/final_comparison.png', dpi=200, bbox_inches='tight')
    
    print("\n✅ Comparison saved to: outputs/final_comparison.png")
    
    # Print summary
    print("\n" + "="*70)
    print("FINAL RESULTS SUMMARY")
    print("="*70)
    print(f"\n{'Model':<25} {'Sensitivity':<15} {'Specificity':<15} {'Accuracy':<15}")
    print("-"*70)
    print(f"{'Old Baseline':<25} {old_metrics['sensitivity']:<15.1%} {old_metrics['specificity']:<15.1%} {old_metrics['accuracy']:<15.1%}")
    print(f"{'Single-Plane Improved':<25} {improved_metrics['sensitivity']:<15.1%} {improved_metrics['specificity']:<15.1%} {improved_metrics['accuracy']:<15.1%}")
    print(f"{'Multi-Plane Fusion':<25} {multiplane_metrics['sensitivity']:<15.1%} {multiplane_metrics['specificity']:<15.1%} {multiplane_metrics['accuracy']:<15.1%}")
    print("="*70)

if __name__ == '__main__':
    main()
