"""
Quick verification that deployment model achieves 90.83%
"""

import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.deploy_medical_model import ProductionMedicalModel
from src.multiplane_loader import MultiPlaneMRNetDataset


def verify_accuracy():
    """Verify we get 90.83% accuracy"""
    
    print("\n" + "="*70)
    print("✅ VERIFYING DEPLOYMENT MODEL ACCURACY")
    print("="*70)
    
    # Load model
    model = ProductionMedicalModel('outputs/resnet_only/best_model.pth')
    
    # Load validation data
    print("\n📊 Loading validation dataset...")
    dataset = MultiPlaneMRNetDataset(
        'dataset/MRNet-v1.0',
        task='abnormal',
        split='valid',
        use_all_slices=False
    )
    
    # Evaluate
    print(f"\n🔍 Evaluating {len(dataset)} cases...")
    all_preds = []
    all_labels = []
    all_probs = []
    
    for idx in tqdm(range(len(dataset))):
        sample = dataset[idx]
        
        result = model.predict(
            sample['sagittal'].unsqueeze(0),
            sample['coronal'].unsqueeze(0),
            sample['axial'].unsqueeze(0)
        )
        
        all_preds.append(result['prediction_class'])
        all_labels.append(sample['label'].item())
        all_probs.append(result['probability'])
    
    # Calculate accuracy
    accuracy = accuracy_score(all_labels, all_preds)
    
    print("\n" + "="*70)
    print("📊 VERIFICATION RESULTS")
    print("="*70)
    print(f"\n✅ Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    if accuracy >= 0.905:
        print("\n🎉 EXCELLENT! Accuracy matches expected 90.83%!")
    elif accuracy >= 0.900:
        print("\n✅ GOOD! Accuracy within acceptable range!")
    else:
        print("\n⚠️  Lower than expected. Should be ~90.83%")
    
    # Confidence stats
    import numpy as np
    probs_array = np.array(all_probs)
    high_conf = sum(1 for p in all_probs if p > 0.8 or p < 0.2)
    
    print(f"\n📊 Confidence Distribution:")
    print(f"   High Confidence: {high_conf}/{len(dataset)} ({high_conf/len(dataset)*100:.1f}%)")
    print(f"   Mean Probability: {probs_array.mean():.3f}")
    print(f"   Std Probability:  {probs_array.std():.3f}")
    
    print("\n" + "="*70)
    print("✅ VERIFICATION COMPLETE!")
    print("="*70)


if __name__ == '__main__':
    verify_accuracy()
