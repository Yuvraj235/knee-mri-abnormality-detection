import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import MRNetDataset
from src.model import ImprovedResNetDeiTFusion

def analyze_prediction_distribution(model, dataset, device):
    model.eval()
    predictions = []
    probabilities = []
    true_labels = []
    case_ids = []
    
    print("\n🔍 Analyzing ALL validation cases...")
    
    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc="Processing"):
            image, label = dataset[idx]
            case_id = str(dataset.labels_df.iloc[idx]['case'])
            
            image_batch = image.unsqueeze(0).to(device)
            output = model(image_batch)
            prob = torch.sigmoid(output).item()
            
            pred = 1 if prob > 0.5 else 0
            
            predictions.append(int(pred))
            probabilities.append(float(prob))
            true_labels.append(int(label))
            case_ids.append(case_id)
    
    results_df = pd.DataFrame({
        'case_id': case_ids,
        'true_label': true_labels,
        'prediction': predictions,
        'probability': probabilities
    })
    
    results_df['true_label'] = results_df['true_label'].astype(int)
    results_df['prediction'] = results_df['prediction'].astype(int)
    results_df['probability'] = results_df['probability'].astype(float)
    
    return results_df

def create_comparison_plot(results_df, output_dir):
    """Create side-by-side comparison of old vs new model"""
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    # Calculate metrics
    cm = confusion_matrix(results_df['true_label'].values, results_df['prediction'].values)
    tn, fp, fn, tp = cm.ravel()
    
    new_sensitivity = tp/(tp+fn) if (tp+fn) > 0 else 0
    new_specificity = tn/(tn+fp) if (tn+fp) > 0 else 0
    new_accuracy = (tp+tn)/len(results_df)
    
    # Old model metrics
    old_sensitivity = 0.958
    old_specificity = 0.400
    old_accuracy = 0.842
    
    # 1. Sensitivity Comparison
    axes[0, 0].bar(['Old Model', 'New Model'], 
                   [old_sensitivity, new_sensitivity],
                   color=['lightcoral', 'lightgreen'])
    axes[0, 0].set_title('Sensitivity (Recall)', fontsize=14, weight='bold')
    axes[0, 0].set_ylabel('Score', fontsize=12)
    axes[0, 0].set_ylim([0, 1])
    axes[0, 0].axhline(y=0.9, color='orange', linestyle='--', alpha=0.5, label='Target >90%')
    for i, v in enumerate([old_sensitivity, new_sensitivity]):
        axes[0, 0].text(i, v, f'{v:.1%}', ha='center', va='bottom', fontsize=12, weight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # 2. Specificity Comparison
    axes[0, 1].bar(['Old Model', 'New Model'], 
                   [old_specificity, new_specificity],
                   color=['red', 'green'])
    axes[0, 1].set_title('Specificity ⭐ KEY IMPROVEMENT', fontsize=14, weight='bold')
    axes[0, 1].set_ylabel('Score', fontsize=12)
    axes[0, 1].set_ylim([0, 1])
    axes[0, 1].axhline(y=0.6, color='orange', linestyle='--', alpha=0.5, label='Target >60%')
    for i, v in enumerate([old_specificity, new_specificity]):
        axes[0, 1].text(i, v, f'{v:.1%}', ha='center', va='bottom', fontsize=12, weight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # 3. Accuracy Comparison
    axes[0, 2].bar(['Old Model', 'New Model'], 
                   [old_accuracy, new_accuracy],
                   color=['lightyellow', 'lightblue'])
    axes[0, 2].set_title('Accuracy', fontsize=14, weight='bold')
    axes[0, 2].set_ylabel('Score', fontsize=12)
    axes[0, 2].set_ylim([0, 1])
    for i, v in enumerate([old_accuracy, new_accuracy]):
        axes[0, 2].text(i, v, f'{v:.1%}', ha='center', va='bottom', fontsize=12, weight='bold')
    axes[0, 2].grid(axis='y', alpha=0.3)
    
    # 4. Confusion Matrix
    axes[1, 0].axis('off')
    cm_text = f"NEW MODEL CONFUSION MATRIX:\n\n"
    cm_text += f"                Predicted\n"
    cm_text += f"              Normal  Abnormal\n"
    cm_text += f"True Normal     {tn:3d}      {fp:3d}\n"
    cm_text += f"     Abnormal   {fn:3d}      {tp:3d}\n\n"
    cm_text += f"True Positives:  {tp} ({tp/(tp+fn)*100:.1f}%)\n"
    cm_text += f"True Negatives:  {tn} ({tn/(tn+fp)*100:.1f}%)\n"
    cm_text += f"False Positives: {fp} ({fp/(tn+fp)*100:.1f}%)\n"
    cm_text += f"False Negatives: {fn} ({fn/(tp+fn)*100:.1f}%)\n"
    
    axes[1, 0].text(0.1, 0.5, cm_text, fontsize=11, verticalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3),
                    family='monospace')
    axes[1, 0].set_title('Confusion Matrix', fontsize=14, weight='bold')
    
    # 5. Key Improvements
    axes[1, 1].axis('off')
    improvement_text = "🎯 KEY IMPROVEMENTS:\n\n"
    improvement_text += f"Specificity:\n"
    improvement_text += f"  40.0% → {new_specificity:.1%}\n"
    improvement_text += f"  +{(new_specificity-old_specificity)*100:.0f} percentage points ✅\n\n"
    improvement_text += f"False Positive Rate:\n"
    improvement_text += f"  60.0% → {fp/(tn+fp)*100:.0f}%\n"
    improvement_text += f"  {(fp/(tn+fp)-0.6)*100:.0f} percentage points ✅\n\n"
    improvement_text += f"TRADEOFF:\n"
    improvement_text += f"Sensitivity:\n"
    improvement_text += f"  95.8% → {new_sensitivity:.1%}\n"
    improvement_text += f"  {(new_sensitivity-old_sensitivity)*100:.0f} percentage points ⚠️\n"
    
    axes[1, 1].text(0.1, 0.5, improvement_text, fontsize=11, verticalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3),
                    family='monospace')
    axes[1, 1].set_title('Performance Gains', fontsize=14, weight='bold')
    
    # 6. Clinical Impact
    axes[1, 2].axis('off')
    clinical_text = "🏥 CLINICAL IMPACT:\n\n"
    clinical_text += "OLD MODEL:\n"
    clinical_text += "• Catches 96% abnormalities ✓\n"
    clinical_text += "• 60% false alarm rate ❌\n"
    clinical_text += "• Overwhelms radiologists\n"
    clinical_text += "• Not suitable for screening\n\n"
    clinical_text += "NEW MODEL:\n"
    clinical_text += f"• Catches {new_sensitivity*100:.0f}% abnormalities\n"
    clinical_text += f"• {fp/(tn+fp)*100:.0f}% false alarm rate ✅\n"
    clinical_text += "• More practical for triage\n"
    clinical_text += "• Better clinical balance\n\n"
    clinical_text += "RECOMMENDATION:\n"
    clinical_text += "Use for initial screening\n"
    clinical_text += "to reduce radiologist workload"
    
    axes[1, 2].text(0.1, 0.5, clinical_text, fontsize=10, verticalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3),
                    family='monospace')
    axes[1, 2].set_title('Clinical Assessment', fontsize=14, weight='bold')
    
    plt.suptitle('OLD vs NEW MODEL COMPARISON', fontsize=18, weight='bold', y=0.98)
    plt.tight_layout()
    
    save_path = f"{output_dir}/model_comparison.png"
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved comparison to {save_path}")

def main():
    print("="*70)
    print("📊 IMPROVED MODEL ANALYSIS")
    print("="*70)
    
    CONFIG = {
        'mrnet_path': '/Users/yuvrajpratapsingh/knee-mri-abnormality-detection/dataset/MRNet-v1.0',
        'model_path': '/Users/yuvrajpratapsingh/knee-mri-abnormality-detection/outputs/models/best_model_improved.pth',
        'output_dir': '/Users/yuvrajpratapsingh/knee-mri-abnormality-detection/outputs/improved_analysis',
        'plane': 'sagittal',
        'task': 'abnormal',
    }
    
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    val_dataset = MRNetDataset(
        root_dir=CONFIG['mrnet_path'],
        plane=CONFIG['plane'],
        task=CONFIG['task'],
        split='valid',
        use_all_slices=False
    )
    
    model = ImprovedResNetDeiTFusion(num_classes=1, fusion_type='concat', pretrained=False, dropout_rate=0.5)
    
    # FIX: Add weights_only=False for PyTorch 2.6+
    checkpoint = torch.load(CONFIG['model_path'], map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    print(f"\n✅ Model loaded successfully!")
    print(f"   Epoch: {checkpoint['epoch']+1}")
    print(f"   Val AUC: {checkpoint['val_auc']:.4f}")
    print(f"   Val Sensitivity: {checkpoint['val_sensitivity']:.4f}")
    print(f"   Val Specificity: {checkpoint['val_specificity']:.4f}")
    
    results_df = analyze_prediction_distribution(model, val_dataset, device)
    
    results_df.to_csv(f"{CONFIG['output_dir']}/predictions_improved.csv", index=False)
    print(f"✅ Saved predictions to CSV")
    
    create_comparison_plot(results_df, CONFIG['output_dir'])
    
    # Print summary
    cm = confusion_matrix(results_df['true_label'].values, results_df['prediction'].values)
    tn, fp, fn, tp = cm.ravel()
    
    sensitivity = tp/(tp+fn) if (tp+fn) > 0 else 0
    specificity = tn/(tn+fp) if (tn+fp) > 0 else 0
    accuracy = (tp+tn)/len(results_df)
    
    print("\n" + "="*70)
    print("🎉 FINAL RESULTS - IMPROVED MODEL")
    print("="*70)
    print(f"\nConfusion Matrix:")
    print(f"  True Positives:  {tp:3d}")
    print(f"  True Negatives:  {tn:3d}")
    print(f"  False Positives: {fp:3d}")
    print(f"  False Negatives: {fn:3d}")
    print(f"\nPerformance:")
    print(f"  Sensitivity: {sensitivity:.1%} (was 95.8%)")
    print(f"  Specificity: {specificity:.1%} (was 40.0%) 🎯")
    print(f"  Accuracy:    {accuracy:.1%} (was 84.2%)")
    print("\n" + "="*70)
    print("✅ Analysis complete!")
    print(f"📁 Check: {CONFIG['output_dir']}/")
    print("="*70)

if __name__ == '__main__':
    main()
