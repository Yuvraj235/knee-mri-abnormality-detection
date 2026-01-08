"""
DEMO: Test model and SAVE results to file
"""

import sys
sys.path.append('.')
from src.deploy_medical_model import ProductionMedicalModel
from src.multiplane_loader import MultiPlaneMRNetDataset
import random
import os
from datetime import datetime


def demo_with_save():
    """Demo that saves results to file"""
    
    print("\n" + "="*70)
    print("🎬 PRODUCTION MODEL DEMO - WITH SAVE")
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
    results_text = []
    
    # Header
    results_text.append("="*70)
    results_text.append("PRODUCTION MODEL DEMO RESULTS")
    results_text.append("="*70)
    results_text.append(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    results_text.append(f"Model: 90.83% Accuracy | Temperature: 0.6617")
    results_text.append("="*70)
    results_text.append("")
    
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
        
        # Print to console
        print(f"\nCase {i+1}/10: {emoji}")
        print(f"  True:        {true_label}")
        print(f"  Predicted:   {result['prediction_label']}")
        print(f"  Probability: {result['probability']:.1%}")
        print(f"  Confidence:  {result['confidence']:.1%}")
        print(f"  {result['recommendation']}")
        
        # Save to text
        results_text.append(f"Case {i+1}/10: {emoji}")
        results_text.append(f"  Validation Index: {idx}")
        results_text.append(f"  True Label:      {true_label}")
        results_text.append(f"  Predicted:       {result['prediction_label']}")
        results_text.append(f"  Probability:     {result['probability']:.4f} ({result['probability']:.1%})")
        results_text.append(f"  Confidence:      {result['confidence']:.4f} ({result['confidence']:.1%})")
        results_text.append(f"  Recommendation:  {result['recommendation']}")
        results_text.append("")
    
    # Summary
    accuracy_pct = correct * 10
    print("\n" + "="*70)
    print(f"📊 Results: {correct}/10 correct ({accuracy_pct}%)")
    print("="*70)
    
    results_text.append("="*70)
    results_text.append(f"SUMMARY: {correct}/10 correct ({accuracy_pct}%)")
    results_text.append("="*70)
    results_text.append("")
    results_text.append("Expected Performance (Full Validation Set):")
    results_text.append("  Accuracy:    90.83%")
    results_text.append("  AUC-ROC:     0.903")
    results_text.append("  Precision:   93.75%")
    results_text.append("  Recall:      94.74%")
    results_text.append("  Specificity: 76.00%")
    results_text.append("")
    results_text.append("="*70)
    results_text.append("Model ready for deployment! 🚀")
    results_text.append("="*70)
    
    # Save to file
    os.makedirs('outputs/demo_results', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = f'outputs/demo_results/demo_{timestamp}.txt'
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(results_text))
    
    print(f"\n✅ Results saved to: {output_path}")
    
    # Also save latest
    latest_path = 'outputs/demo_results/latest_demo.txt'
    with open(latest_path, 'w') as f:
        f.write('\n'.join(results_text))
    
    print(f"✅ Also saved as: {latest_path}")
    
    print("\n📁 All saved files:")
    print("   📊 outputs/medical_system_evaluation.png")
    print("   📄 outputs/MEDICAL_SYSTEM_EVALUATION_REPORT.txt")
    print("   🎯 outputs/demo_results/latest_demo.txt")
    print("   🏆 outputs/resnet_only/best_model.pth")


if __name__ == '__main__':
    demo_with_save()
