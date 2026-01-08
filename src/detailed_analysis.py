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
from src.model import ResNetDeiTFusion

def analyze_prediction_distribution(model, dataset, device):
    """Analyze what the model is actually predicting"""
    
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
            
            predictions.append(int(pred))  # Ensure int
            probabilities.append(float(prob))  # Ensure float
            true_labels.append(int(label))  # Ensure int
            case_ids.append(case_id)
    
    results_df = pd.DataFrame({
        'case_id': case_ids,
        'true_label': true_labels,
        'prediction': predictions,
        'probability': probabilities
    })
    
    # Ensure proper dtypes
    results_df['true_label'] = results_df['true_label'].astype(int)
    results_df['prediction'] = results_df['prediction'].astype(int)
    results_df['probability'] = results_df['probability'].astype(float)
    
    return results_df

def create_comprehensive_report(results_df, output_dir):
    """Create detailed analysis report"""
    
    fig = plt.figure(figsize=(24, 16))
    gs = fig.add_gridspec(4, 4, hspace=0.35, wspace=0.35)
    
    # 1. Prediction Distribution
    ax1 = fig.add_subplot(gs[0, 0])
    pred_counts = results_df['prediction'].value_counts().sort_index()
    labels_pred = ['Normal' if k == 0 else 'Abnormal' for k in pred_counts.index]
    
    ax1.bar(labels_pred, pred_counts.values, color=['lightgreen', 'lightcoral'])
    ax1.set_title('Model Predictions Distribution', fontsize=14, weight='bold')
    ax1.set_ylabel('Count', fontsize=12)
    ax1.grid(axis='y', alpha=0.3)
    
    total = len(results_df)
    for i, (label, count) in enumerate(zip(labels_pred, pred_counts.values)):
        ax1.text(i, count, f'{count}\n({count/total*100:.1f}%)', 
                ha='center', va='bottom', fontsize=11, weight='bold')
    
    # 2. True Label Distribution
    ax2 = fig.add_subplot(gs[0, 1])
    true_counts = results_df['true_label'].value_counts().sort_index()
    labels_true = ['Normal' if k == 0 else 'Abnormal' for k in true_counts.index]
    
    ax2.bar(labels_true, true_counts.values, color=['lightgreen', 'lightcoral'], alpha=0.7)
    ax2.set_title('True Label Distribution', fontsize=14, weight='bold')
    ax2.set_ylabel('Count', fontsize=12)
    ax2.grid(axis='y', alpha=0.3)
    
    for i, (label, count) in enumerate(zip(labels_true, true_counts.values)):
        ax2.text(i, count, f'{count}\n({count/total*100:.1f}%)', 
                ha='center', va='bottom', fontsize=11, weight='bold')
    
    # 3. Probability Distribution by True Label
    ax3 = fig.add_subplot(gs[0, 2:])
    normal_probs = results_df[results_df['true_label'] == 0]['probability']
    abnormal_probs = results_df[results_df['true_label'] == 1]['probability']
    
    ax3.hist(normal_probs, bins=20, alpha=0.6, label='True Normal', color='green', edgecolor='black')
    ax3.hist(abnormal_probs, bins=20, alpha=0.6, label='True Abnormal', color='red', edgecolor='black')
    ax3.axvline(0.5, color='black', linestyle='--', linewidth=2, label='Threshold')
    ax3.set_xlabel('Predicted Probability (Abnormal)', fontsize=12, weight='bold')
    ax3.set_ylabel('Count', fontsize=12, weight='bold')
    ax3.set_title('Probability Distribution by True Label', fontsize=14, weight='bold')
    ax3.legend(fontsize=11)
    ax3.grid(alpha=0.3)
    
    # 4. Confusion Matrix
    ax4 = fig.add_subplot(gs[1, :2])
    cm = confusion_matrix(results_df['true_label'].values, results_df['prediction'].values)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn_r', ax=ax4,
                xticklabels=['Normal', 'Abnormal'],
                yticklabels=['Normal', 'Abnormal'],
                annot_kws={'size': 16, 'weight': 'bold'},
                cbar_kws={'label': 'Count'})
    ax4.set_xlabel('Predicted Label', fontsize=12, weight='bold')
    ax4.set_ylabel('True Label', fontsize=12, weight='bold')
    ax4.set_title('Confusion Matrix', fontsize=14, weight='bold')
    
    # Add percentages
    for i in range(2):
        for j in range(2):
            if cm[i].sum() > 0:
                percent = cm[i, j] / cm[i].sum() * 100
                ax4.text(j+0.5, i+0.7, f'({percent:.1f}%)', 
                        ha='center', va='center', fontsize=11, color='blue')
    
    # 5. Performance Metrics Table
    ax5 = fig.add_subplot(gs[1, 2:])
    ax5.axis('off')
    
    tn, fp, fn, tp = cm.ravel()
    
    sensitivity = tp/(tp+fn) if (tp+fn) > 0 else 0
    specificity = tn/(tn+fp) if (tn+fp) > 0 else 0
    precision = tp/(tp+fp) if (tp+fp) > 0 else 0
    accuracy = (tp+tn)/len(results_df)
    
    metrics = [
        ['Metric', 'Value', 'Interpretation'],
        ['True Positives', f'{tp}', f'{tp/len(results_df)*100:.1f}% of all cases'],
        ['True Negatives', f'{tn}', f'{tn/len(results_df)*100:.1f}% of all cases'],
        ['False Positives', f'{fp}', f'{fp/len(results_df)*100:.1f}% of all cases'],
        ['False Negatives', f'{fn}', f'{fn/len(results_df)*100:.1f}% of all cases'],
        ['', '', ''],
        ['Sensitivity (Recall)', f'{sensitivity:.3f}', 'Catches abnormalities'],
        ['Specificity', f'{specificity:.3f}', 'Identifies normals'],
        ['Precision', f'{precision:.3f}', 'Abnormal predictions correct'],
        ['Accuracy', f'{accuracy:.3f}', 'Overall correctness'],
    ]
    
    table = ax5.table(cellText=metrics, cellLoc='left', loc='center',
                      colWidths=[0.35, 0.25, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    for i in range(3):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    for i in range(1, len(metrics)):
        for j in range(3):
            if i == 6:
                table[(i, j)].set_facecolor('#ffffff')
            elif i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    ax5.set_title('Performance Metrics', fontsize=14, weight='bold', pad=20)
    
    # 6. Prediction Confidence Analysis
    ax6 = fig.add_subplot(gs[2, :2])
    
    correct_df = results_df[results_df['prediction'] == results_df['true_label']]
    incorrect_df = results_df[results_df['prediction'] != results_df['true_label']]
    
    correct_conf = correct_df['probability'].apply(lambda x: x if x > 0.5 else 1-x)
    incorrect_conf = incorrect_df['probability'].apply(lambda x: x if x > 0.5 else 1-x)
    
    ax6.hist(correct_conf, bins=20, alpha=0.7, label='Correct Predictions', 
            color='green', edgecolor='black')
    ax6.hist(incorrect_conf, bins=20, alpha=0.7, label='Incorrect Predictions',
            color='red', edgecolor='black')
    ax6.set_xlabel('Confidence', fontsize=12, weight='bold')
    ax6.set_ylabel('Count', fontsize=12, weight='bold')
    ax6.set_title('Prediction Confidence by Correctness', fontsize=14, weight='bold')
    ax6.legend(fontsize=11)
    ax6.grid(alpha=0.3)
    
    # 7. Error Analysis
    ax7 = fig.add_subplot(gs[2, 2:])
    ax7.axis('off')
    
    fp_cases = results_df[(results_df['true_label'] == 0) & (results_df['prediction'] == 1)]
    fn_cases = results_df[(results_df['true_label'] == 1) & (results_df['prediction'] == 0)]
    
    error_text = "ERROR ANALYSIS:\n\n"
    error_text += f"False Positives (Normal → Predicted Abnormal): {len(fp_cases)}\n"
    if len(fp_cases) > 0:
        avg_fp_conf = fp_cases['probability'].mean()
        error_text += f"  Average Confidence: {avg_fp_conf:.1%}\n"
        error_text += f"  Cases: {', '.join(fp_cases['case_id'].head(10).tolist())}\n\n"
    
    error_text += f"False Negatives (Abnormal → Predicted Normal): {len(fn_cases)}\n"
    if len(fn_cases) > 0:
        avg_fn_conf = (1 - fn_cases['probability']).mean()
        error_text += f"  Average Confidence: {avg_fn_conf:.1%}\n"
        error_text += f"  Cases: {', '.join(fn_cases['case_id'].head(10).tolist())}\n\n"
    
    error_text += "\n⚠️ KEY FINDINGS:\n"
    
    pred_abnormal_rate = (results_df['prediction'] == 1).sum() / total
    if pred_abnormal_rate > 0.9:
        error_text += f"• SEVERE BIAS: Predicts Abnormal {pred_abnormal_rate*100:.1f}% of time\n"
    
    if len(fp_cases) > len(fn_cases) * 2:
        error_text += "• High False Positive Rate (over-diagnosing)\n"
    
    if len(fp_cases) > 0 and fp_cases['probability'].mean() > 0.8:
        error_text += "• Overconfident on False Positives\n"
    
    if specificity < 0.5:
        error_text += f"• Poor Specificity ({specificity:.1%}): Can't identify normals\n"
    
    ax7.text(0.05, 0.95, error_text, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
            family='monospace')
    ax7.set_title('Detailed Error Analysis', fontsize=14, weight='bold')
    
    # 8. Worst Cases
    ax8 = fig.add_subplot(gs[3, :])
    ax8.axis('off')
    
    if len(incorrect_df) > 0:
        incorrect_sorted = incorrect_df.sort_values('probability', ascending=False)
        
        worst_text = "MOST CONFIDENT MISTAKES:\n\n"
        worst_text += f"{'Case':<12} {'True':<10} {'Pred':<10} {'Prob':<8} {'Conf':<8}\n"
        worst_text += "-" * 60 + "\n"
        
        for idx, row in incorrect_sorted.head(15).iterrows():
            conf = row['probability'] if row['probability'] > 0.5 else 1 - row['probability']
            true_label = 'Normal' if row['true_label'] == 0 else 'Abnormal'
            pred_label = 'Normal' if row['prediction'] == 0 else 'Abnormal'
            
            worst_text += f"{row['case_id']:<12} {true_label:<10} {pred_label:<10} "
            worst_text += f"{row['probability']:.3f}    {conf:.1%}\n"
        
        ax8.text(0.05, 0.95, worst_text, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.3),
                family='monospace')
        ax8.set_title('Most Confident Mistakes', fontsize=14, weight='bold')
    
    fig.suptitle('COMPREHENSIVE MODEL ANALYSIS', fontsize=20, weight='bold', y=0.995)
    
    save_path = f"{output_dir}/comprehensive_model_analysis.png"
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Saved comprehensive analysis to {save_path}")

def main():
    print("="*70)
    print("📊 DETAILED MODEL ANALYSIS - FULL VALIDATION SET")
    print("="*70)
    
    CONFIG = {
        'mrnet_path': '/Users/yuvrajpratapsingh/knee-mri-abnormality-detection/dataset/MRNet-v1.0',
        'model_path': '/Users/yuvrajpratapsingh/knee-mri-abnormality-detection/outputs/models/best_model.pth',
        'output_dir': '/Users/yuvrajpratapsingh/knee-mri-abnormality-detection/outputs/detailed_analysis',
        'plane': 'sagittal',
        'task': 'abnormal',
    }
    
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\n🖥️  Device: {device}")
    
    # Load dataset
    print("\n📦 Loading validation dataset...")
    val_dataset = MRNetDataset(
        root_dir=CONFIG['mrnet_path'],
        plane=CONFIG['plane'],
        task=CONFIG['task'],
        split='valid',
        use_all_slices=False
    )
    print(f"✅ Loaded {len(val_dataset)} cases")
    
    # Load model
    print("\n🤖 Loading model...")
    model = ResNetDeiTFusion(num_classes=1, fusion_type='concat', pretrained=False)
    checkpoint = torch.load(CONFIG['model_path'], map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    print(f"✅ Model loaded (AUC: {checkpoint['val_auc']:.4f})")
    
    # Analyze
    results_df = analyze_prediction_distribution(model, val_dataset, device)
    
    # Save detailed results
    results_df.to_csv(f"{CONFIG['output_dir']}/all_predictions.csv", index=False)
    print(f"✅ Saved all predictions to all_predictions.csv")
    
    # Create report
    print("\n📊 Creating comprehensive analysis report...")
    create_comprehensive_report(results_df, CONFIG['output_dir'])
    
    # Print summary
    print("\n" + "="*70)
    print("📋 QUICK SUMMARY")
    print("="*70)
    
    pred_counts = results_df['prediction'].value_counts().sort_index()
    print(f"\nPredictions:")
    print(f"  Normal:   {pred_counts.get(0, 0):3d} ({pred_counts.get(0, 0)/len(results_df)*100:5.1f}%)")
    print(f"  Abnormal: {pred_counts.get(1, 0):3d} ({pred_counts.get(1, 0)/len(results_df)*100:5.1f}%)")
    
    true_counts = results_df['true_label'].value_counts().sort_index()
    print(f"\nTrue Labels:")
    print(f"  Normal:   {true_counts.get(0, 0):3d} ({true_counts.get(0, 0)/len(results_df)*100:5.1f}%)")
    print(f"  Abnormal: {true_counts.get(1, 0):3d} ({true_counts.get(1, 0)/len(results_df)*100:5.1f}%)")
    
    cm = confusion_matrix(results_df['true_label'].values, results_df['prediction'].values)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"\nConfusion Matrix:")
    print(f"  True Positives:  {tp:3d}")
    print(f"  True Negatives:  {tn:3d}")
    print(f"  False Positives: {fp:3d} ⚠️")
    print(f"  False Negatives: {fn:3d} ⚠️")
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    print(f"\nPerformance:")
    print(f"  Sensitivity: {sensitivity:.1%}")
    print(f"  Specificity: {specificity:.1%}")
    print(f"  Accuracy:    {(tp+tn)/len(results_df):.1%}")
    
    print("\n" + "="*70)
    print(f"✅ Complete! Check: {CONFIG['output_dir']}/")
    print("="*70)
    
    # View results
    print(f"\nopen {CONFIG['output_dir']}/comprehensive_model_analysis.png")

if __name__ == '__main__':
    main()
