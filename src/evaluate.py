import os
import sys
import torch
import numpy as np
from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score, 
                             confusion_matrix, classification_report, roc_curve)
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import MRNetDataset
from src.model import ResNetDeiTFusion

def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Evaluating'):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            probs = torch.sigmoid(outputs).cpu().numpy()
            
            all_probs.extend(probs)
            all_preds.extend((probs > 0.5).astype(int).flatten())
            all_labels.extend(labels.cpu().numpy())
    
    return np.array(all_preds), np.array(all_labels), np.array(all_probs).flatten()

def plot_confusion_matrix(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['Normal', 'Abnormal'],
                yticklabels=['Normal', 'Abnormal'],
                annot_kws={'size': 16})
    
    plt.title('Confusion Matrix', fontsize=18, weight='bold', pad=20)
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Confusion matrix saved to {save_path}")
    plt.close()

def plot_roc_curve(y_true, y_probs, auc_score, save_path):
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, linewidth=3, label=f'ROC Curve (AUC = {auc_score:.4f})', color='blue')
    plt.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Random Classifier')
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('ROC Curve', fontsize=18, weight='bold', pad=20)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ ROC curve saved to {save_path}")
    plt.close()

def main():
    CONFIG = {
        'mrnet_path': '/Users/yuvrajpratapsingh/knee-mri-abnormality-detection/dataset/MRNet-v1.0',
        'model_path': '/Users/yuvrajpratapsingh/knee-mri-abnormality-detection/outputs/models/best_model.pth',
        'output_dir': '/Users/yuvrajpratapsingh/knee-mri-abnormality-detection/outputs/plots',
        'plane': 'sagittal',
        'task': 'abnormal',
        'batch_size': 16,
    }
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🖥️  Using device: {device}\n")
    
    # Load dataset
    print("📦 Loading validation dataset...")
    val_dataset = MRNetDataset(
        root_dir=CONFIG['mrnet_path'],
        plane=CONFIG['plane'],
        task=CONFIG['task'],
        split='valid',
        use_all_slices=False
    )
    
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], 
                           shuffle=False, num_workers=2)
    
    # Load model
    print("\n🤖 Loading best model...")
    model = ResNetDeiTFusion(num_classes=1, fusion_type='concat', pretrained=False)
    checkpoint = torch.load(CONFIG['model_path'], map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    print(f"✅ Model loaded from epoch {checkpoint['epoch'] + 1}")
    print(f"✅ Saved validation AUC: {checkpoint['val_auc']:.4f}\n")
    
    # Evaluate
    print("📊 Evaluating model on validation set...")
    y_pred, y_true, y_probs = evaluate_model(model, val_loader, device)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_probs)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn)  # Recall
    specificity = tn / (tn + fp)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    # Print results
    print("\n" + "="*70)
    print("🏆 FINAL EVALUATION RESULTS")
    print("="*70)
    print(f"📊 Accuracy:     {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"📊 AUC-ROC:      {auc:.4f} ({auc*100:.2f}%)")
    print(f"📊 F1 Score:     {f1:.4f}")
    print(f"📊 Precision:    {precision:.4f}")
    print(f"📊 Sensitivity:  {sensitivity:.4f} (Recall)")
    print(f"📊 Specificity:  {specificity:.4f}")
    print("="*70)
    
    print("\n📋 Confusion Matrix:")
    print(f"   True Negatives:  {tn} (Correctly predicted Normal)")
    print(f"   False Positives: {fp} (Normal predicted as Abnormal)")
    print(f"   False Negatives: {fn} (Abnormal predicted as Normal)")
    print(f"   True Positives:  {tp} (Correctly predicted Abnormal)")
    
    print("\n📋 Detailed Classification Report:")
    print(classification_report(y_true, y_pred, 
                                target_names=['Normal', 'Abnormal'],
                                digits=4))
    
    # Save visualizations
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    
    cm_path = f"{CONFIG['output_dir']}/confusion_matrix_final.png"
    plot_confusion_matrix(y_true, y_pred, cm_path)
    
    roc_path = f"{CONFIG['output_dir']}/roc_curve_final.png"
    plot_roc_curve(y_true, y_probs, auc, roc_path)
    
    # Summary comparison
    print("\n" + "="*70)
    print("📊 COMPARISON WITH BASELINE")
    print("="*70)
    print(f"Original MRNet Paper AUC:  ~0.8500 (85%)")
    print(f"Your Model AUC:             {auc:.4f} ({auc*100:.2f}%)")
    print(f"Improvement:               +{(auc - 0.85)*100:.2f}%")
    print("="*70)
    
    print("\n✅ Evaluation complete!")
    print(f"📊 Visualizations saved to {CONFIG['output_dir']}/")
    print(f"\n🎉 Congratulations! Your model achieved {auc*100:.2f}% AUC!")

if __name__ == '__main__':
    main()
