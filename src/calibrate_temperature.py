"""
Temperature Calibration for Medical Model
Fixes probability outputs to match actual accuracy
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import log_loss
from tqdm import tqdm
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.multiplane_loader import MultiPlaneMRNetDataset
from src.multiplane_model import MultiPlaneFusion


def calibrate_temperature(model_path, dataset_path, device='mps'):
    """
    Calibrate temperature parameter on validation set
    """
    
    print("\n" + "="*70)
    print("🌡️  TEMPERATURE CALIBRATION")
    print("="*70)
    
    device = torch.device(device if torch.backends.mps.is_available() else 'cpu')
    
    # Load model
    print("\n📦 Loading model...")
    model = MultiPlaneFusion(num_classes=1, dropout_rate=0.4)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Load validation data
    print("📊 Loading validation data...")
    val_dataset = MultiPlaneMRNetDataset(
        dataset_path,
        task='abnormal',
        split='valid',
        use_all_slices=False
    )
    
    # Collect logits and labels
    print("🔍 Collecting predictions...")
    all_logits = []
    all_labels = []
    
    with torch.no_grad():
        for idx in tqdm(range(len(val_dataset))):
            sample = val_dataset[idx]
            
            sag = sample['sagittal'].unsqueeze(0).to(device)
            cor = sample['coronal'].unsqueeze(0).to(device)
            axi = sample['axial'].unsqueeze(0).to(device)
            label = sample['label'].item()
            
            logits = model(sag, cor, axi)
            
            all_logits.append(logits.item())
            all_labels.append(label)
    
    all_logits = torch.tensor(all_logits).to(device)
    all_labels = torch.tensor(all_labels).to(device)
    
    # Optimize temperature
    print("\n🌡️  Finding optimal temperature...")
    temperature = nn.Parameter(torch.ones(1).to(device))
    optimizer = optim.LBFGS([temperature], lr=0.01, max_iter=50)
    
    def eval_loss():
        optimizer.zero_grad()
        scaled_logits = all_logits / temperature
        loss = nn.BCEWithLogitsLoss()(scaled_logits, all_labels.float())
        loss.backward()
        return loss
    
    optimizer.step(eval_loss)
    
    optimal_temp = temperature.item()
    
    # Evaluate before and after
    with torch.no_grad():
        # Before calibration
        probs_before = torch.sigmoid(all_logits).cpu().numpy()
        
        # After calibration
        probs_after = torch.sigmoid(all_logits / temperature).cpu().numpy()
    
    labels_np = all_labels.cpu().numpy()
    
    from sklearn.metrics import accuracy_score
    
    preds_before = (probs_before > 0.5).astype(int)
    preds_after = (probs_after > 0.5).astype(int)
    
    acc_before = accuracy_score(labels_np, preds_before)
    acc_after = accuracy_score(labels_np, preds_after)
    
    print("\n" + "="*70)
    print("📊 CALIBRATION RESULTS")
    print("="*70)
    print(f"\n🌡️  Optimal Temperature: {optimal_temp:.4f}")
    print(f"\n📈 Accuracy:")
    print(f"   Before: {acc_before:.4f} ({acc_before*100:.2f}%)")
    print(f"   After:  {acc_after:.4f} ({acc_after*100:.2f}%)")
    
    print(f"\n📊 Probability Distribution:")
    print(f"   Before - Mean: {probs_before.mean():.3f}, Std: {probs_before.std():.3f}")
    print(f"   After  - Mean: {probs_after.mean():.3f}, Std: {probs_after.std():.3f}")
    
    print("\n" + "="*70)
    print(f"✅ CALIBRATION COMPLETE!")
    print(f"💾 Optimal temperature: {optimal_temp:.4f}")
    print("="*70)
    
    return optimal_temp


if __name__ == '__main__':
    model_path = '/Users/yuvrajpratapsingh/knee-mri-abnormality-detection/outputs/resnet_only/best_model.pth'
    dataset_path = '/Users/yuvrajpratapsingh/knee-mri-abnormality-detection/dataset/MRNet-v1.0'
    
    optimal_temp = calibrate_temperature(model_path, dataset_path)
    
    # Save temperature
    import json
    with open('outputs/optimal_temperature.json', 'w') as f:
        json.dump({'temperature': optimal_temp}, f)
    
    print(f"\n💾 Saved to: outputs/optimal_temperature.json")
    print(f"Use this temperature value in your medical model!")
