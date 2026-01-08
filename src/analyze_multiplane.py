import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.multiplane_loader import MultiPlaneMRNetDataset
from src.multiplane_model import MultiPlaneFusion

def main():
    print("="*70)
    print("📊 MULTI-PLANE MODEL ANALYSIS")
    print("="*70)
    
    CONFIG = {
        'mrnet_path': '/Users/yuvrajpratapsingh/knee-mri-abnormality-detection/dataset/MRNet-v1.0',
        'model_path': '/Users/yuvrajpratapsingh/knee-mri-abnormality-detection/outputs/models/best_model_multiplane.pth',
        'output_dir': '/Users/yuvrajpratapsingh/knee-mri-abnormality-detection/outputs/multiplane_analysis',
        'task': 'abnormal',
    }
    
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Load dataset
    val_dataset = MultiPlaneMRNetDataset(
        root_dir=CONFIG['mrnet_path'],
        task=CONFIG['task'],
        split='valid',
        use_all_slices=False
    )
    
    # Load model
    model = MultiPlaneFusion(num_classes=1, dropout_rate=0.4)
    # FIX: Add weights_only=False
    checkpoint = torch.load(CONFIG['model_path'], map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"\n✅ Model loaded from epoch {checkpoint['epoch']+1}")
    print(f"   Val AUC: {checkpoint['val_auc']:.4f}")
    print(f"   Val Sensitivity: {checkpoint['val_sensitivity']:.4f}")
    print(f"   Val Specificity: {checkpoint['val_specificity']:.4f}")
    
    # Analyze
    predictions = []
    probabilities = []
    true_labels = []
    
    print("\n🔍 Analyzing validation set...")
    with torch.no_grad():
        for idx in tqdm(range(len(val_dataset))):
            batch = val_dataset[idx]
            
            sagittal = batch['sagittal'].unsqueeze(0).to(device)
            coronal = batch['coronal'].unsqueeze(0).to(device)
            axial = batch['axial'].unsqueeze(0).to(device)
            label = batch['label'].item()
            
            output = model(sagittal, coronal, axial)
            prob = torch.sigmoid(output).item()
            pred = 1 if prob > 0.5 else 0
            
            predictions.append(pred)
            probabilities.append(prob)
            true_labels.append(label)
    
    # Results
    results_df = pd.DataFrame({
        'true_label': true_labels,
        'prediction': predictions,
        'probability': probabilities
    })
    
    results_df.to_csv(f"{CONFIG['output_dir']}/predictions_multiplane.csv", index=False)
    
    # Metrics
    cm = confusion_matrix(true_labels, predictions)
    tn, fp, fn, tp = cm.ravel()
    
    sensitivity = tp/(tp+fn) if (tp+fn) > 0 else 0
    specificity = tn/(tn+fp) if (tn+fp) > 0 else 0
    accuracy = (tp+tn)/len(true_labels)
    
    print("\n" + "="*70)
    print("🎉 MULTI-PLANE MODEL RESULTS")
    print("="*70)
    print(f"\nConfusion Matrix:")
    print(f"  TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
    print(f"\nPerformance:")
    print(f"  Sensitivity: {sensitivity:.1%}")
    print(f"  Specificity: {specificity:.1%}")
    print(f"  Accuracy:    {accuracy:.1%}")
    print("="*70)
    print(f"\n✅ Saved predictions to: {CONFIG['output_dir']}/predictions_multiplane.csv")

if __name__ == '__main__':
    main()
