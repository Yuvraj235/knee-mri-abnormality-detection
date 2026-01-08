import matplotlib.pyplot as plt
import numpy as np

def create_comparison_chart():
    """Create visual comparison of models"""
    
    # Data (will be updated with actual results)
    models = ['Baseline\n(Original)', 'Hybrid\n(Too Complex)', 'ResNet-Only\n(New)']
    accuracy = [87.5, 79.2, 88.0]  # Placeholder - will update
    auc = [90.5, 82.0, 91.0]  # Placeholder
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Accuracy comparison
    bars1 = ax1.bar(models, accuracy, color=['#3498db', '#e74c3c', '#2ecc71'])
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylim([70, 95])
    ax1.axhline(y=87.5, color='red', linestyle='--', alpha=0.5, label='Baseline')
    ax1.legend()
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontweight='bold')
    
    # AUC comparison
    bars2 = ax2.bar(models, auc, color=['#3498db', '#e74c3c', '#2ecc71'])
    ax2.set_ylabel('AUC-ROC (%)', fontsize=12)
    ax2.set_title('Model AUC-ROC Comparison', fontsize=14, fontweight='bold')
    ax2.set_ylim([70, 95])
    ax2.axhline(y=90.5, color='red', linestyle='--', alpha=0.5, label='Baseline')
    ax2.legend()
    
    # Add value labels
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('outputs/model_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: outputs/model_comparison.png")
    plt.show()

if __name__ == '__main__':
    create_comparison_chart()
