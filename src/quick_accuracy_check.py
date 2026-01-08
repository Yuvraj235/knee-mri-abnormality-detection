"""
Quick accuracy check without Monte Carlo dropout
"""

import torch
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.multiplane_loader import MultiPlaneMRNetDataset
from src.multiplane_model import MultiPlaneFusion


def quick_evaluation():
    print("\n" + "="*70)
    print("⚡ QUICK ACCURACY CHECK (No Monte Carlo)")
    print("="*70)
    
    model_path = '/Users/yuvrajpratapsingh/knee-mri-abnormality-detection/outputs/resnet_only/best_model.pth'
    dataset_path = '/Users/yuvrajpratapsingh/knee-mri-abnormality-detection/dataset/MRNet-v1.0'
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Load model
    print("\n📦 Loading model...")
    model = MultiPlaneFusion(num_classes=1, dropout_rate=0.4)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Load dataset
    print("📊 Loading validation data...")
    dataset = MultiPlaneMRNetDataset(
        dataset_path,
        task='abnormal',
        split='valid',
        use_all_slices=False
    )
    
    print(f"🔍 Evaluating {len(dataset)} cases...\n")
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for idx in tqdm(range(len(dataset))):
            sample = dataset[idx]
            
            sag = sample['sagittal'].unsqueeze(0).to(device)
            cor = sample['coronal'].unsqueeze(0).to(device)
            axi = sample['axial'].unsqueeze(0).to(device)
            label = sample['label'].item()
            
            logits = model(sag, cor, axi)
            prob = torch.sigmoid(logits).item()
            pred = 1 if prob > 0.5 else 0
            
            all_preds.append(pred)
            all_labels.append(label)
            all_probs.append(prob)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    
    # Probability statistics
    probs_array = np.array(all_probs)
    high_conf = sum(1 for p in all_probs if p > 0.8 or p < 0.2)
    medium_conf = sum(1 for p in all_probs if 0.6 <= p <= 0.8 or 0.2 <= p <= 0.4)
    low_conf = sum(1 for p in all_probs if 0.4 < p < 0.6)
    
    print("\n" + "="*70)
    print("📊 RESULTS")
    print("="*70)
    print(f"\n✅ Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"✅ AUC-ROC:  {auc:.4f}")
    
    print(f"\n📊 Probability Statistics:")
    print(f"   Mean: {probs_array.mean():.3f}")
    print(f"   Std:  {probs_array.std():.3f}")
    print(f"   Min:  {probs_array.min():.3f}")
    print(f"   Max:  {probs_array.max():.3f}")
    
    print(f"\n🎯 Confidence Distribution:")
    print(f"   High Confidence (>80% or <20%):   {high_conf:3d} ({high_conf/len(dataset)*100:.1f}%)")
    print(f"   Medium Confidence (60-80%, 20-40%): {medium_conf:3d} ({medium_conf/len(dataset)*100:.1f}%)")
    print(f"   Low Confidence (40-60%):          {low_conf:3d} ({low_conf/len(dataset)*100:.1f}%)")
    
    print("\n" + "="*70)
    
    # Show some example predictions
    print("\n�� Sample Predictions:")
    for i in range(min(5, len(dataset))):
        print(f"   Case {i+1}: True={all_labels[i]}, Pred={all_preds[i]}, Prob={all_probs[i]:.3f}")


if __name__ == '__main__':
    quick_evaluation()
