import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import MRNetDataset
from src.model import ResNetDeiTFusion

def analyze_errors(model, dataset, device):
    """Analyze misclassified cases"""
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    all_cases = []
    
    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc="Analyzing"):
            image, label = dataset[idx]
            case_id = dataset.labels_df.iloc[idx]['case']
            
            image = image.unsqueeze(0).to(device)
            output = model(image)
            prob = torch.sigmoid(output).item()
            
            all_probs.append(prob)
            all_preds.append(1 if prob > 0.5 else 0)
            all_labels.append(label)
            all_cases.append(case_id)
    
    # Create DataFrame
    df = pd.DataFrame({
        'case_id': all_cases,
        'true_label': all_labels,
        'prediction': all_preds,
        'probability': all_probs,
        'correct': np.array(all_preds) == np.array(all_labels)
    })
    
    # Identify errors
    false_positives = df[(df['true_label'] == 0) & (df['prediction'] == 1)]
    false_negatives = df[(df['true_label'] == 1) & (df['prediction'] == 0)]
    
    return df, false_positives, false_negatives

def create_analysis_plots(df, output_dir):
    """Create comprehensive analysis plots"""
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Probability distribution
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(df[df['true_label']==0]['probability'], bins=20, alpha=0.5, label='Normal', color='green')
    ax1.hist(df[df['true_label']==1]['probability'], bins=20, alpha=0.5, label='Abnormal', color='red')
    ax1.axvline(0.5, color='black', linestyle='--', linewidth=2, label='Threshold')
    ax1.set_xlabel('Predicted Probability', fontsize=12, weight='bold')
    ax1.set_ylabel('Count', fontsize=12, weight='bold')
    ax1.set_title('Probability Distribution', fontsize=14, weight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # 2. Confidence vs Correctness
    ax2 = fig.add_subplot(gs[0, 1])
    correct_df = df[df['correct'] == True]
    incorrect_df = df[df['correct'] == False]
    
    correct_conf = correct_df['probability'].apply(lambda x: x if x > 0.5 else 1-x)
    incorrect_conf = incorrect_df['probability'].apply(lambda x: x if x > 0.5 else 1-x)
    
    ax2.hist(correct_conf, bins=20, alpha=0.6, label='Correct', color='green')
    ax2.hist(incorrect_conf, bins=20, alpha=0.6, label='Incorrect', color='red')
    ax2.set_xlabel('Confidence', fontsize=12, weight='bold')
    ax2.set_ylabel('Count', fontsize=12, weight='bold')
    ax2.set_title('Confidence Distribution', fontsize=14, weight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # 3. Error types pie chart
    ax3 = fig.add_subplot(gs[0, 2])
    tp = len(df[(df['true_label']==1) & (df['prediction']==1)])
    tn = len(df[(df['true_label']==0) & (df['prediction']==0)])
    fp = len(df[(df['true_label']==0) & (df['prediction']==1)])
    fn = len(df[(df['true_label']==1) & (df['prediction']==0)])
    
    sizes = [tp, tn, fp, fn]
    labels = [f'True Pos\n({tp})', f'True Neg\n({tn})', 
              f'False Pos\n({fp})', f'False Neg\n({fn})']
    colors = ['lightgreen', 'lightblue', 'lightyellow', 'lightcoral']
    
    ax3.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax3.set_title('Prediction Breakdown', fontsize=14, weight='bold')
    
    # 4. ROC Curve
    ax4 = fig.add_subplot(gs[1, :2])
    fpr, tpr, _ = roc_curve(df['true_label'], df['probability'])
    roc_auc = auc(fpr, tpr)
    
    ax4.plot(fpr, tpr, linewidth=3, label=f'ROC Curve (AUC = {roc_auc:.4f})', color='blue')
    ax4.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Random')
    ax4.set_xlabel('False Positive Rate', fontsize=12, weight='bold')
    ax4.set_ylabel('True Positive Rate', fontsize=12, weight='bold')
    ax4.set_title('ROC Curve', fontsize=14, weight='bold')
    ax4.legend(fontsize=11)
    ax4.grid(alpha=0.3)
    
    # 5. Confusion Matrix
    ax5 = fig.add_subplot(gs[1, 2])
    cm = confusion_matrix(df['true_label'], df['prediction'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax5,
                xticklabels=['Normal', 'Abnormal'],
                yticklabels=['Normal', 'Abnormal'],
                annot_kws={'size': 14, 'weight': 'bold'})
    ax5.set_xlabel('Predicted', fontsize=12, weight='bold')
    ax5.set_ylabel('True', fontsize=12, weight='bold')
    ax5.set_title('Confusion Matrix', fontsize=14, weight='bold')
    
    # 6. Probability calibration
    ax6 = fig.add_subplot(gs[2, 0])
    bins = np.linspace(0, 1, 11)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    prob_true = []
    prob_pred = []
    
    for i in range(len(bins)-1):
        mask = (df['probability'] >= bins[i]) & (df['probability'] < bins[i+1])
        if mask.sum() > 0:
            prob_pred.append(df[mask]['probability'].mean())
            prob_true.append(df[mask]['true_label'].mean())
    
    ax6.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    ax6.plot(prob_pred, prob_true, 'o-', linewidth=2, markersize=8, label='Model')
    ax6.set_xlabel('Predicted Probability', fontsize=12, weight='bold')
    ax6.set_ylabel('True Probability', fontsize=12, weight='bold')
    ax6.set_title('Calibration Curve', fontsize=14, weight='bold')
    ax6.legend()
    ax6.grid(alpha=0.3)
    
    # 7. Statistics table
    ax7 = fig.add_subplot(gs[2, 1:])
    ax7.axis('off')
    
    stats = [
        ['Metric', 'Value'],
        ['Total Cases', f"{len(df)}"],
        ['Correct Predictions', f"{df['correct'].sum()} ({df['correct'].mean()*100:.1f}%)"],
        ['True Positives', f"{tp}"],
        ['True Negatives', f"{tn}"],
        ['False Positives', f"{fp}"],
        ['False Negatives', f"{fn}"],
        ['Sensitivity (Recall)', f"{tp/(tp+fn):.3f}"],
        ['Specificity', f"{tn/(tn+fp):.3f}"],
        ['Precision', f"{tp/(tp+fp):.3f}"],
        ['F1 Score', f"{2*tp/(2*tp+fp+fn):.3f}"],
        ['AUC', f"{roc_auc:.4f}"],
    ]
    
    table = ax7.table(cellText=stats, cellLoc='left', loc='center',
                      colWidths=[0.4, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)
    
    # Style header
    for i in range(2):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(stats)):
        for j in range(2):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    fig.suptitle('Comprehensive Model Analysis', fontsize=18, weight='bold', y=0.98)
    
    save_path = f"{output_dir}/comprehensive_analysis.png"
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved comprehensive analysis to {save_path}")

def main():
    print("="*70)
    print("📊 COMPREHENSIVE MODEL ANALYSIS")
    print("="*70)
    
    CONFIG = {
        'mrnet_path': '/Users/yuvrajpratapsingh/knee-mri-abnormality-detection/dataset/MRNet-v1.0',
        'model_path': '/Users/yuvrajpratapsingh/knee-mri-abnormality-detection/outputs/models/best_model.pth',
        'output_dir': '/Users/yuvrajpratapsingh/knee-mri-abnormality-detection/outputs/analysis',
        'plane': 'sagittal',
        'task': 'abnormal',
    }
    
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\n🖥️  Using device: {device}")
    
    # Load dataset
    print("\n📦 Loading validation dataset...")
    val_dataset = MRNetDataset(
        root_dir=CONFIG['mrnet_path'],
        plane=CONFIG['plane'],
        task=CONFIG['task'],
        split='valid',
        use_all_slices=False
    )
    
    # Load model
    print("\n🤖 Loading model...")
    model = ResNetDeiTFusion(num_classes=1, fusion_type='concat', pretrained=False)
    checkpoint = torch.load(CONFIG['model_path'], map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    # Analyze
    print("\n🔍 Analyzing predictions...")
    df, false_positives, false_negatives = analyze_errors(model, val_dataset, device)
    
    # Save results
    df.to_csv(f"{CONFIG['output_dir']}/detailed_predictions.csv", index=False)
    print(f"✅ Saved detailed predictions to detailed_predictions.csv")
    
    # Print error analysis
    print("\n" + "="*70)
    print("⚠️  ERROR ANALYSIS")
    print("="*70)
    print(f"\nFalse Positives: {len(false_positives)}")
    if len(false_positives) > 0:
        print("  Cases:", false_positives['case_id'].tolist())
    
    print(f"\nFalse Negatives: {len(false_negatives)}")
    if len(false_negatives) > 0:
        print("  Cases:", false_negatives['case_id'].tolist())
    
    # Create plots
    print("\n📊 Creating analysis plots...")
    create_analysis_plots(df, CONFIG['output_dir'])
    
    print("\n✅ Analysis complete!")

if __name__ == '__main__':
    main()
