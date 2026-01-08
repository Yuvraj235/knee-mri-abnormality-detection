import matplotlib.pyplot as plt
import numpy as np

def create_comprehensive_visualization():
    """Create publication-quality comparison charts"""
    
    # Data
    models = ['Baseline\n(87.5%)', 'Hybrid\n(79.2%)', 'ResNet-Only\n(90.8%)']
    accuracy = [87.5, 79.2, 90.8]
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    
    # Main accuracy comparison
    ax1 = plt.subplot(2, 2, 1)
    bars = ax1.bar(models, accuracy, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    ax1.set_ylabel('Validation Accuracy (%)', fontsize=14, fontweight='bold')
    ax1.set_title('Model Performance Comparison', fontsize=16, fontweight='bold')
    ax1.set_ylim([70, 95])
    ax1.axhline(y=87.5, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Baseline')
    ax1.grid(axis='y', alpha=0.3)
    ax1.legend(fontsize=12)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}%',
                ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    # Improvement over baseline
    ax2 = plt.subplot(2, 2, 2)
    improvements = [0, -8.3, +3.3]
    bar_colors = ['gray', '#e74c3c', '#2ecc71']
    bars2 = ax2.bar(models, improvements, color=bar_colors, alpha=0.8, edgecolor='black', linewidth=2)
    ax2.set_ylabel('Accuracy Change (%)', fontsize=14, fontweight='bold')
    ax2.set_title('Improvement vs Baseline', fontsize=16, fontweight='bold')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars2, improvements):
        if val >= 0:
            va = 'bottom'
            offset = 0.3
        else:
            va = 'top'
            offset = -0.3
        ax2.text(bar.get_x() + bar.get_width()/2., val + offset,
                f'{val:+.1f}%',
                ha='center', va=va, fontweight='bold', fontsize=12)
    
    # Model complexity
    ax3 = plt.subplot(2, 2, 3)
    params = [76, 90, 76]  # Million parameters
    model_names_short = ['Baseline', 'Hybrid', 'ResNet-Only']
    bars3 = ax3.bar(model_names_short, params, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    ax3.set_ylabel('Parameters (Millions)', fontsize=14, fontweight='bold')
    ax3.set_title('Model Complexity', fontsize=16, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    
    for bar in bars3:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{int(height)}M',
                ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # Training epochs comparison
    ax4 = plt.subplot(2, 2, 4)
    epochs = [25, 20, 25]
    bars4 = ax4.bar(model_names_short, epochs, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    ax4.set_ylabel('Training Epochs', fontsize=14, fontweight='bold')
    ax4.set_title('Training Configuration', fontsize=16, fontweight='bold')
    ax4.set_ylim([0, 30])
    ax4.grid(axis='y', alpha=0.3)
    
    for bar in bars4:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{int(height)}',
                ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # Add overall title
    fig.suptitle('Knee MRI Abnormality Detection - Final Results', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    # Add footer
    fig.text(0.5, 0.02, 
             '🏆 Winner: ResNet-Only Model (90.83% Accuracy) | +3.33% Improvement over Baseline',
             ha='center', fontsize=14, fontweight='bold', 
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig('outputs/FINAL_COMPARISON.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: outputs/FINAL_COMPARISON.png")
    plt.show()

if __name__ == '__main__':
    create_comprehensive_visualization()
