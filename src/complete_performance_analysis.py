import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score, f1_score
)
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.multiplane_loader import MultiPlaneMRNetDataset
from src.multiplane_model import MultiPlaneFusion

def evaluate_complete_performance(model, dataset, device):
    """Complete performance evaluation"""
    
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    print("📊 Evaluating all cases...")
    with torch.no_grad():
        for idx in tqdm(range(len(dataset))):
            batch = dataset[idx]
            
            sagittal = batch['sagittal'].unsqueeze(0).to(device)
            coronal = batch['coronal'].unsqueeze(0).to(device)
            axial = batch['axial'].unsqueeze(0).to(device)
            label = batch['label'].item()
            
            output = model(sagittal, coronal, axial)
            prob = torch.sigmoid(output).item()
            pred = 1 if prob > 0.5 else 0
            
            all_predictions.append(pred)
            all_labels.append(label)
            all_probabilities.append(prob)
    
    return np.array(all_predictions), np.array(all_labels), np.array(all_probabilities)

def create_comprehensive_visualizations(y_true, y_pred, y_prob, output_dir):
    """Create all performance visualizations"""
    
    fig = plt.figure(figsize=(24, 20))
    gs = fig.add_gridspec(5, 4, hspace=0.4, wspace=0.3)
    
    # Title
    fig.suptitle('COMPLETE MODEL PERFORMANCE ANALYSIS', fontsize=20, weight='bold')
    
    # 1. Confusion Matrix (Large)
    ax1 = fig.add_subplot(gs[0:2, 0:2])
    cm = confusion_matrix(y_true, y_pred)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=['Normal', 'Abnormal'],
                yticklabels=['Normal', 'Abnormal'],
                cbar_kws={'label': 'Count'},
                annot_kws={'size': 16})
    
    ax1.set_xlabel('Predicted', fontsize=14, weight='bold')
    ax1.set_ylabel('True', fontsize=14, weight='bold')
    ax1.set_title('Confusion Matrix', fontsize=16, weight='bold')
    
    # Add percentages
    tn, fp, fn, tp = cm.ravel()
    total = tn + fp + fn + tp
    
    # Add text annotations with percentages
    ax1.text(0.5, 0.25, f'{tn/total:.1%}', ha='center', va='center',
             color='gray', fontsize=12, transform=ax1.transAxes)
    ax1.text(1.5, 0.25, f'{fp/total:.1%}', ha='center', va='center',
             color='gray', fontsize=12, transform=ax1.transAxes)
    ax1.text(0.5, 0.75, f'{fn/total:.1%}', ha='center', va='center',
             color='gray', fontsize=12, transform=ax1.transAxes)
    ax1.text(1.5, 0.75, f'{tp/total:.1%}', ha='center', va='center',
             color='gray', fontsize=12, transform=ax1.transAxes)
    
    # 2. ROC Curve
    ax2 = fig.add_subplot(gs[0, 2])
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    ax2.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Chance')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('False Positive Rate', fontsize=11, weight='bold')
    ax2.set_ylabel('True Positive Rate', fontsize=11, weight='bold')
    ax2.set_title('ROC Curve', fontsize=12, weight='bold')
    ax2.legend(loc="lower right", fontsize=9)
    ax2.grid(alpha=0.3)
    
    # 3. Precision-Recall Curve
    ax3 = fig.add_subplot(gs[0, 3])
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    avg_precision = average_precision_score(y_true, y_prob)
    
    ax3.plot(recall, precision, color='blue', lw=2, label=f'AP = {avg_precision:.3f}')
    ax3.set_xlim([0.0, 1.0])
    ax3.set_ylim([0.0, 1.05])
    ax3.set_xlabel('Recall', fontsize=11, weight='bold')
    ax3.set_ylabel('Precision', fontsize=11, weight='bold')
    ax3.set_title('Precision-Recall Curve', fontsize=12, weight='bold')
    ax3.legend(loc="lower left", fontsize=9)
    ax3.grid(alpha=0.3)
    
    # 4. Probability Distribution
    ax4 = fig.add_subplot(gs[1, 2])
    
    # Separate by true label
    prob_abnormal = y_prob[y_true == 1]
    prob_normal = y_prob[y_true == 0]
    
    ax4.hist(prob_normal, bins=20, alpha=0.7, label='True Normal', color='blue', edgecolor='black')
    ax4.hist(prob_abnormal, bins=20, alpha=0.7, label='True Abnormal', color='red', edgecolor='black')
    ax4.axvline(0.5, color='black', linestyle='--', linewidth=2, label='Threshold')
    ax4.set_xlabel('Predicted Probability', fontsize=11, weight='bold')
    ax4.set_ylabel('Count', fontsize=11, weight='bold')
    ax4.set_title('Probability Distribution by True Label', fontsize=12, weight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(alpha=0.3)
    
    # 5. Confidence Distribution
    ax5 = fig.add_subplot(gs[1, 3])
    confidence = np.abs(y_prob - 0.5) * 2
    
    # Separate by correctness
    correct_mask = (y_pred == y_true)
    conf_correct = confidence[correct_mask]
    conf_incorrect = confidence[~correct_mask]
    
    ax5.hist(conf_correct, bins=20, alpha=0.7, label='Correct', color='green', edgecolor='black')
    ax5.hist(conf_incorrect, bins=20, alpha=0.7, label='Incorrect', color='red', edgecolor='black')
    ax5.set_xlabel('Confidence', fontsize=11, weight='bold')
    ax5.set_ylabel('Count', fontsize=11, weight='bold')
    ax5.set_title('Confidence Distribution', fontsize=12, weight='bold')
    ax5.legend(fontsize=9)
    ax5.grid(alpha=0.3)
    
    # 6. Metrics by Threshold
    ax6 = fig.add_subplot(gs[2, :2])
    thresholds_range = np.linspace(0, 1, 100)
    
    sensitivities = []
    specificities = []
    accuracies = []
    f1_scores = []
    
    for thresh in thresholds_range:
        pred_at_thresh = (y_prob >= thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred_at_thresh).ravel()
        
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        acc = (tp + tn) / (tp + tn + fp + fn)
        f1 = f1_score(y_true, pred_at_thresh)
        
        sensitivities.append(sens)
        specificities.append(spec)
        accuracies.append(acc)
        f1_scores.append(f1)
    
    ax6.plot(thresholds_range, sensitivities, label='Sensitivity', linewidth=2)
    ax6.plot(thresholds_range, specificities, label='Specificity', linewidth=2)
    ax6.plot(thresholds_range, accuracies, label='Accuracy', linewidth=2)
    ax6.plot(thresholds_range, f1_scores, label='F1 Score', linewidth=2)
    ax6.axvline(0.5, color='red', linestyle='--', linewidth=2, label='Current Threshold')
    ax6.set_xlabel('Threshold', fontsize=11, weight='bold')
    ax6.set_ylabel('Score', fontsize=11, weight='bold')
    ax6.set_title('Performance Metrics vs Threshold', fontsize=12, weight='bold')
    ax6.legend(fontsize=9)
    ax6.grid(alpha=0.3)
    ax6.set_ylim([0, 1])
    
    # 7. Calibration Plot
    ax7 = fig.add_subplot(gs[2, 2:])
    
    # Bin predictions
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    bin_sums = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)
    
    for i in range(n_bins):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i+1])
        if mask.sum() > 0:
            bin_sums[i] = y_true[mask].sum()
            bin_counts[i] = mask.sum()
    
    bin_true = np.divide(bin_sums, bin_counts, where=bin_counts>0)
    
    ax7.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    ax7.plot(bin_centers, bin_true, 'o-', linewidth=2, markersize=8, label='Model')
    ax7.set_xlabel('Predicted Probability', fontsize=11, weight='bold')
    ax7.set_ylabel('True Probability', fontsize=11, weight='bold')
    ax7.set_title('Calibration Plot', fontsize=12, weight='bold')
    ax7.legend(fontsize=9)
    ax7.grid(alpha=0.3)
    ax7.set_xlim([0, 1])
    ax7.set_ylim([0, 1])
    
    # 8. Performance Summary Table
    ax8 = fig.add_subplot(gs[3:, :2])
    ax8.axis('off')
    
    # Calculate all metrics
    tn, fp, fn, tp = cm.ravel()
    
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    f1 = f1_score(y_true, y_pred)
    
    summary_text = f"""
╔════════════════════════════════════════════════════════╗
║          COMPREHENSIVE PERFORMANCE METRICS             ║
╠════════════════════════════════════════════════════════╣
║                                                        ║
║  CONFUSION MATRIX:                                     ║
║    True Positives (TP):     {tp:3d}  ({tp/(tp+fn)*100:5.1f}% of abnormal)  ║
║    True Negatives (TN):     {tn:3d}  ({tn/(tn+fp)*100:5.1f}% of normal)    ║
║    False Positives (FP):    {fp:3d}  ({fp/(tn+fp)*100:5.1f}% of normal)    ║
║    False Negatives (FN):    {fn:3d}  ({fn/(tp+fn)*100:5.1f}% of abnormal)  ║
║                                                        ║
║  PRIMARY METRICS:                                      ║
║    Sensitivity (Recall):    {sensitivity*100:5.1f}%                     ║
║    Specificity:             {specificity*100:5.1f}%                     ║
║    Accuracy:                {accuracy*100:5.1f}%                     ║
║    F1 Score:                {f1:5.3f}                       ║
║                                                        ║
║  PREDICTIVE VALUES:                                    ║
║    Positive Predictive Value (Precision): {ppv*100:5.1f}%      ║
║    Negative Predictive Value:             {npv*100:5.1f}%      ║
║                                                        ║
║  DISCRIMINATION:                                       ║
║    AUC-ROC:                 {roc_auc:5.3f}                       ║
║    Average Precision:       {avg_precision:5.3f}                       ║
║                                                        ║
║  CONFIDENCE ANALYSIS:                                  ║
║    Avg Confidence (Correct):   {conf_correct.mean()*100:5.1f}%          ║
║    Avg Confidence (Incorrect): {conf_incorrect.mean()*100:5.1f}%          ║
║                                                        ║
║  CLINICAL INTERPRETATION:                              ║
║    - Out of 100 abnormal cases, model detects {sensitivity*100:.0f}      ║
║    - Out of 100 normal cases, model correctly          ║
║      identifies {specificity*100:.0f} as normal                         ║
║    - When model says "abnormal", it's right {ppv*100:.0f}% of time ║
║    - When model says "normal", it's right {npv*100:.0f}% of time  ║
║                                                        ║
╚════════════════════════════════════════════════════════╝
"""
    
    ax8.text(0.1, 0.5, summary_text, fontsize=9, verticalalignment='center',
            family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    # 9. Error Analysis
    ax9 = fig.add_subplot(gs[3:, 2:])
    ax9.axis('off')
    
    # Analyze errors
    fp_probs = y_prob[(y_pred == 1) & (y_true == 0)]
    fn_probs = y_prob[(y_pred == 0) & (y_true == 1)]
    
    error_text = f"""
╔═══════════════════════════════════════════════╗
║            ERROR ANALYSIS                     ║
╠═══════════════════════════════════════════════╣
║                                               ║
║  FALSE POSITIVES (n={fp}):                       ║
║    Avg Probability: {fp_probs.mean()*100 if len(fp_probs)>0 else 0:5.1f}%              ║
║    Min Probability: {fp_probs.min()*100 if len(fp_probs)>0 else 0:5.1f}%              ║
║    Max Probability: {fp_probs.max()*100 if len(fp_probs)>0 else 0:5.1f}%              ║
║                                               ║
║    Interpretation:                            ║
║    - Model flagged {fp} normal cases as abnormal ║
║    - {fp/(tn+fp)*100:.1f}% false alarm rate                 ║
║    - Avg confidence: {np.abs(fp_probs-0.5).mean()*200 if len(fp_probs)>0 else 0:.1f}%                ║
║                                               ║
║  FALSE NEGATIVES (n={fn}):                       ║
║    Avg Probability: {fn_probs.mean()*100 if len(fn_probs)>0 else 0:5.1f}%              ║
║    Min Probability: {fn_probs.min()*100 if len(fn_probs)>0 else 0:5.1f}%              ║
║    Max Probability: {fn_probs.max()*100 if len(fn_probs)>0 else 0:5.1f}%              ║
║                                               ║
║    Interpretation:                            ║
║    - Model missed {fn} abnormal cases            ║
║    - {fn/(tp+fn)*100:.1f}% miss rate                        ║
║    - Avg confidence: {np.abs(fn_probs-0.5).mean()*200 if len(fn_probs)>0 else 0:.1f}%                ║
║                                               ║
║  KEY INSIGHTS:                                ║
║    • High confidence on errors suggests        ║
║      {('hard cases' if conf_incorrect.mean() > 0.5 else 'model uncertainty')}                               ║
║    • {'FP' if fp > fn else 'FN'} is the dominant error type           ║
║    • {'Conservative' if fp < fn else 'Aggressive'} model behavior               ║
║                                               ║
║  RECOMMENDATIONS:                             ║
║    • All predictions with confidence < 50%    ║
║      should undergo radiologist review        ║
║    • {'Consider adjusting threshold' if abs(fp-fn) > 5 else 'Current threshold is balanced'}    ║
║                                               ║
╚═══════════════════════════════════════════════╝
"""
    
    ax9.text(0.1, 0.5, error_text, fontsize=9, verticalalignment='center',
            family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.6))
    
    plt.savefig(f'{output_dir}/complete_performance_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved: complete_performance_analysis.png")

def generate_detailed_report(y_true, y_pred, y_prob, output_dir):
    """Generate detailed text report"""
    
    report_path = f'{output_dir}/COMPLETE_PERFORMANCE_REPORT.txt'
    
    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("COMPLETE MODEL PERFORMANCE REPORT\n")
        f.write("Multi-Plane Fusion Model for Knee MRI Abnormality Detection\n")
        f.write("="*70 + "\n\n")
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        f.write("CONFUSION MATRIX:\n")
        f.write("-" * 40 + "\n")
        f.write(f"                Predicted Negative  Predicted Positive\n")
        f.write(f"Actual Negative        {tn:4d}                {fp:4d}\n")
        f.write(f"Actual Positive        {fn:4d}                {tp:4d}\n\n")
        
        # Classification Report
        f.write("CLASSIFICATION REPORT:\n")
        f.write("-" * 40 + "\n")
        f.write(classification_report(y_true, y_pred, 
                                     target_names=['Normal', 'Abnormal'],
                                     digits=4))
        f.write("\n")
        
        # Additional Metrics
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        avg_precision = average_precision_score(y_true, y_prob)
        
        f.write("ADDITIONAL PERFORMANCE METRICS:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Sensitivity (Recall):           {sensitivity:.4f} ({sensitivity*100:.2f}%)\n")
        f.write(f"Specificity:                    {specificity:.4f} ({specificity*100:.2f}%)\n")
        f.write(f"Positive Predictive Value:      {ppv:.4f} ({ppv*100:.2f}%)\n")
        f.write(f"Negative Predictive Value:      {npv:.4f} ({npv*100:.2f}%)\n")
        f.write(f"AUC-ROC:                        {roc_auc:.4f}\n")
        f.write(f"Average Precision:              {avg_precision:.4f}\n\n")
        
        # Confidence Analysis
        confidence = np.abs(y_prob - 0.5) * 2
        correct_mask = (y_pred == y_true)
        
        f.write("CONFIDENCE ANALYSIS:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Average Confidence (All):       {confidence.mean():.4f} ({confidence.mean()*100:.2f}%)\n")
        f.write(f"Average Confidence (Correct):   {confidence[correct_mask].mean():.4f} ({confidence[correct_mask].mean()*100:.2f}%)\n")
        f.write(f"Average Confidence (Incorrect): {confidence[~correct_mask].mean():.4f} ({confidence[~correct_mask].mean()*100:.2f}%)\n\n")
        
        # Clinical Interpretation
        f.write("CLINICAL INTERPRETATION:\n")
        f.write("-" * 40 + "\n")
        f.write(f"For every 100 abnormal knees:\n")
        f.write(f"  - {tp} will be correctly identified ({sensitivity*100:.1f}%)\n")
        f.write(f"  - {fn} will be missed ({(fn/(tp+fn))*100:.1f}%)\n\n")
        
        f.write(f"For every 100 normal knees:\n")
        f.write(f"  - {tn} will be correctly identified ({specificity*100:.1f}%)\n")
        f.write(f"  - {fp} will be false alarms ({(fp/(tn+fp))*100:.1f}%)\n\n")
        
        f.write(f"When the model predicts 'Abnormal':\n")
        f.write(f"  - It is correct {ppv*100:.1f}% of the time\n\n")
        
        f.write(f"When the model predicts 'Normal':\n")
        f.write(f"  - It is correct {npv*100:.1f}% of the time\n\n")
        
        f.write("="*70 + "\n")
    
    print(f"✅ Saved: COMPLETE_PERFORMANCE_REPORT.txt")

def main():
    print("="*70)
    print("📊 COMPLETE MODEL PERFORMANCE ANALYSIS")
    print("="*70)
    
    CONFIG = {
        'mrnet_path': '/Users/yuvrajpratapsingh/knee-mri-abnormality-detection/dataset/MRNet-v1.0',
        'model_path': '/Users/yuvrajpratapsingh/knee-mri-abnormality-detection/outputs/models/best_model_multiplane.pth',
        'output_dir': '/Users/yuvrajpratapsingh/knee-mri-abnormality-detection/outputs/final_analysis',
        'task': 'abnormal',
    }
    
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\n🖥️  Using device: {device}")
    
    # Load dataset
    print("\n📦 Loading validation dataset...")
    val_dataset = MultiPlaneMRNetDataset(
        root_dir=CONFIG['mrnet_path'],
        task=CONFIG['task'],
        split='valid',
        use_all_slices=False
    )
    
    # Load model
    print("\n🤖 Loading model...")
    model = MultiPlaneFusion(num_classes=1, dropout_rate=0.4)
    checkpoint = torch.load(CONFIG['model_path'], map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    print(f"✅ Model loaded from epoch {checkpoint['epoch']+1}")
    
    # Evaluate
    y_pred, y_true, y_prob = evaluate_complete_performance(model, val_dataset, device)
    
    # Create visualizations
    print("\n📊 Creating comprehensive visualizations...")
    create_comprehensive_visualizations(y_true, y_pred, y_prob, CONFIG['output_dir'])
    
    # Generate report
    print("\n📄 Generating detailed report...")
    generate_detailed_report(y_true, y_pred, y_prob, CONFIG['output_dir'])
    
    # Save predictions
    results_df = pd.DataFrame({
        'case_id': [val_dataset.labels_df.iloc[i]['case'] for i in range(len(val_dataset))],
        'true_label': y_true,
        'predicted_label': y_pred,
        'probability': y_prob,
        'confidence': np.abs(y_prob - 0.5) * 2,
        'correct': y_pred == y_true
    })
    results_df.to_csv(f"{CONFIG['output_dir']}/all_predictions.csv", index=False)
    
    print("\n" + "="*70)
    print("✅ COMPLETE PERFORMANCE ANALYSIS FINISHED!")
    print("="*70)
    print(f"\n📁 Results saved to: {CONFIG['output_dir']}/")
    print(f"   - complete_performance_analysis.png")
    print(f"   - COMPLETE_PERFORMANCE_REPORT.txt")
    print(f"   - all_predictions.csv")
    print("="*70)

if __name__ == '__main__':
    main()
