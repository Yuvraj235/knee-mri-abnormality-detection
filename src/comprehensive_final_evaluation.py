import os
import sys
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, confusion_matrix, classification_report
)
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.multiplane_loader import MultiPlaneMRNetDataset
from src.multiplane_model import MultiPlaneFusion

def evaluate_model(model_path, dataset_path, model_name="Model"):
    """Comprehensive model evaluation"""
    
    print("="*70)
    print(f"📊 EVALUATING: {model_name}")
    print("="*70)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Load validation dataset
    val_dataset = MultiPlaneMRNetDataset(
        root_dir=dataset_path,
        task='abnormal',
        split='valid',
        use_all_slices=False
    )
    
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    # Load model
    model = MultiPlaneFusion(num_classes=1, dropout_rate=0.4)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    print("\n🔍 Running inference...")
    with torch.no_grad():
        for batch in val_loader:
            sag = batch['sagittal'].to(device)
            cor = batch['coronal'].to(device)
            axi = batch['axial'].to(device)
            label = batch['label'].item()
            
            output = model(sag, cor, axi)
            prob = torch.sigmoid(output).item()
            pred = 1 if prob > 0.5 else 0
            
            all_preds.append(pred)
            all_labels.append(label)
            all_probs.append(prob)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    
    # Calculate specificity
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # Print results
    print("\n" + "="*70)
    print(f"RESULTS: {model_name}")
    print("="*70)
    print(f"\n📈 Overall Metrics:")
    print(f"   Accuracy:    {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   AUC-ROC:     {auc:.4f}")
    print(f"   Precision:   {precision:.4f}")
    print(f"   Recall:      {recall:.4f} (Sensitivity)")
    print(f"   Specificity: {specificity:.4f}")
    print(f"   F1-Score:    {f1:.4f}")
    
    print(f"\n📊 Confusion Matrix:")
    print(f"   True Negatives:  {tn:3d}")
    print(f"   False Positives: {fp:3d}")
    print(f"   False Negatives: {fn:3d}")
    print(f"   True Positives:  {tp:3d}")
    
    print(f"\n�� Classification Report:")
    print(classification_report(all_labels, all_preds, 
                                target_names=['Normal', 'Abnormal'],
                                digits=4))
    
    print("="*70)
    
    return {
        'accuracy': accuracy,
        'auc': auc,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1': f1,
        'confusion_matrix': cm
    }


def compare_models():
    """Compare baseline vs new model"""
    
    print("\n" + "="*70)
    print("🔬 MODEL COMPARISON")
    print("="*70)
    
    dataset_path = '/Users/yuvrajpratapsingh/knee-mri-abnormality-detection/dataset/MRNet-v1.0'
    
    # Baseline model (your original best)
    baseline_path = '/Users/yuvrajpratapsingh/knee-mri-abnormality-detection/outputs/mrnet_results/models/best_model_multiplane.pth'
    
    # New model (ResNet-only with potential pre-training)
    new_model_path = '/Users/yuvrajpratapsingh/knee-mri-abnormality-detection/outputs/resnet_only/best_model.pth'
    
    print("\n📊 Evaluating BASELINE model...")
    if os.path.exists(baseline_path):
        baseline_results = evaluate_model(baseline_path, dataset_path, "BASELINE (Original)")
    else:
        print("⚠️  Baseline model not found, skipping comparison")
        baseline_results = None
    
    print("\n�� Evaluating NEW model...")
    if os.path.exists(new_model_path):
        new_results = evaluate_model(new_model_path, dataset_path, "NEW (ResNet-Only)")
    else:
        print("❌ New model not found yet - still training?")
        return
    
    # Comparison
    if baseline_results:
        print("\n" + "="*70)
        print("📊 COMPARISON SUMMARY")
        print("="*70)
        
        print(f"\n{'Metric':<15} {'Baseline':<12} {'New Model':<12} {'Change':<12}")
        print("-"*70)
        
        metrics = ['accuracy', 'auc', 'precision', 'recall', 'specificity', 'f1']
        for metric in metrics:
            baseline_val = baseline_results[metric]
            new_val = new_results[metric]
            change = new_val - baseline_val
            change_str = f"{'+' if change >= 0 else ''}{change:.4f}"
            emoji = "✅" if change >= 0 else "⚠️"
            
            print(f"{metric:<15} {baseline_val:<12.4f} {new_val:<12.4f} {change_str:<12} {emoji}")
        
        print("="*70)
        
        # Verdict
        if new_results['accuracy'] > baseline_results['accuracy']:
            print("\n🎉 NEW MODEL IS BETTER!")
            improvement = (new_results['accuracy'] - baseline_results['accuracy']) * 100
            print(f"   Improvement: +{improvement:.2f}%")
        elif new_results['accuracy'] == baseline_results['accuracy']:
            print("\n✅ NEW MODEL MATCHES BASELINE")
        else:
            print("\n⚠️  BASELINE IS STILL BETTER")
            decline = (baseline_results['accuracy'] - new_results['accuracy']) * 100
            print(f"   Decline: -{decline:.2f}%")


if __name__ == '__main__':
    compare_models()
