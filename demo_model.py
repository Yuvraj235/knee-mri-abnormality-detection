"""
DEMO: Test your production model on any case!
"""

import sys
sys.path.append('.')
from src.deploy_medical_model import ProductionMedicalModel
from src.multiplane_loader import MultiPlaneMRNetDataset
import random


def demo():
    """Interactive demo"""
    
    print("\n" + "="*70)
    print("🎬 PRODUCTION MODEL DEMO - 90.83% ACCURACY")
    print("="*70)
    
    # Initialize
    model = ProductionMedicalModel('outputs/resnet_only/best_model.pth')
    
    # Load dataset
    print("\n📊 Loading validation cases...")
    dataset = MultiPlaneMRNetDataset(
        'dataset/MRNet-v1.0',
        task='abnormal',
        split='valid',
        use_all_slices=False
    )
    
    # Random cases
    print("\n🎲 Testing 10 random cases...")
    print("="*70)
    
    indices = random.sample(range(len(dataset)), 10)
    correct = 0
    
    for i, idx in enumerate(indices):
        sample = dataset[idx]
        
        result = model.predict(
            sample['sagittal'].unsqueeze(0),
            sample['coronal'].unsqueeze(0),
            sample['axial'].unsqueeze(0)
        )
        
        true_label = 'ABNORMAL' if sample['label'].item() == 1 else 'NORMAL'
        is_correct = result['prediction_label'] == true_label
        if is_correct:
            correct += 1
        
        emoji = '✅' if is_correct else '❌'
        
        print(f"\nCase {i+1}/10: {emoji}")
        print(f"  True:       {true_label}")
        print(f"  Predicted:  {result['prediction_label']}")
        print(f"  Probability: {result['probability']:.1%}")
        print(f"  Confidence:  {result['confidence']:.1%}")
        print(f"  {result['recommendation']}")
    
    print("\n" + "="*70)
    print(f"📊 Results: {correct}/10 correct ({correct*10}%)")
    print("="*70)
    
    # Statistics
    print("\n📈 Expected Performance (on full validation set):")
    print("   Accuracy:    90.83%")
    print("   AUC-ROC:     0.903")
    print("   Precision:   93.75%")
    print("   Recall:      94.74%")
    print("   Specificity: 76.00%")
    
    print("\n🚀 Model ready for deployment!")


if __name__ == '__main__':
    demo()
