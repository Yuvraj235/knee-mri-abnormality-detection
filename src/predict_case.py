import os
import sys
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.multiplane_loader import MultiPlaneMRNetDataset
from src.multiplane_model import MultiPlaneFusion

def predict_single_case(model, dataset, case_id, device):
    """
    Predict and explain a single case
    
    Args:
        model: Trained model
        dataset: Dataset
        case_id: Case ID to predict
        device: Device to run on
    """
    
    # Find case index
    case_idx = dataset.labels_df[dataset.labels_df['case'] == case_id].index[0]
    
    # Get data
    batch = dataset[case_idx]
    true_label = batch['label'].item()
    
    sagittal = batch['sagittal'].unsqueeze(0).to(device)
    coronal = batch['coronal'].unsqueeze(0).to(device)
    axial = batch['axial'].unsqueeze(0).to(device)
    
    # Predict
    model.eval()
    with torch.no_grad():
        output = model(sagittal, coronal, axial)
        prob = torch.sigmoid(output).item()
        pred = 1 if prob > 0.5 else 0
    
    # Create visualization
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Denormalize images
    def denormalize(tensor):
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = tensor.cpu() * std + mean
        tensor = torch.clamp(tensor, 0, 1)
        return (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    
    # Show images
    axes[0].imshow(denormalize(batch['sagittal']))
    axes[0].set_title('Sagittal', fontsize=14, weight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(denormalize(batch['coronal']))
    axes[1].set_title('Coronal', fontsize=14, weight='bold')
    axes[1].axis('off')
    
    axes[2].imshow(denormalize(batch['axial']))
    axes[2].set_title('Axial', fontsize=14, weight='bold')
    axes[2].axis('off')
    
    # Prediction panel
    axes[3].axis('off')
    
    # Color based on correctness
    if pred == true_label:
        result_color = 'green'
        result_text = "✅ CORRECT"
    else:
        result_color = 'red'
        result_text = "❌ INCORRECT"
    
    pred_text = f"CASE {case_id:04d}\n\n"
    pred_text += f"{'='*30}\n"
    pred_text += f"TRUE LABEL:\n"
    pred_text += f"  {'ABNORMAL' if true_label == 1 else 'NORMAL'}\n\n"
    pred_text += f"PREDICTION:\n"
    pred_text += f"  {'ABNORMAL' if pred == 1 else 'NORMAL'}\n"
    pred_text += f"  Probability: {prob:.1%}\n"
    pred_text += f"  Confidence: {abs(prob-0.5)*2:.1%}\n\n"
    pred_text += f"RESULT: {result_text}\n"
    pred_text += f"{'='*30}\n\n"
    
    if prob > 0.7:
        pred_text += "High confidence ABNORMAL\n"
        pred_text += "→ Priority review"
    elif prob < 0.3:
        pred_text += "High confidence NORMAL\n"
        pred_text += "→ Standard queue"
    else:
        pred_text += "Low confidence prediction\n"
        pred_text += "→ Radiologist review"
    
    axes[3].text(0.1, 0.5, pred_text, fontsize=12, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7),
                family='monospace')
    
    plt.suptitle(f'PREDICTION FOR CASE {case_id:04d}', fontsize=16, weight='bold', color=result_color)
    plt.tight_layout()
    plt.show()
    
    # Print detailed results
    print("\n" + "="*70)
    print(f"PREDICTION RESULTS FOR CASE {case_id:04d}")
    print("="*70)
    print(f"\nTrue Label:      {'ABNORMAL' if true_label == 1 else 'NORMAL'}")
    print(f"Prediction:      {'ABNORMAL' if pred == 1 else 'NORMAL'}")
    print(f"Probability:     {prob:.4f} ({prob:.1%})")
    print(f"Confidence:      {abs(prob-0.5)*2:.4f} ({abs(prob-0.5)*2:.1%})")
    print(f"\nResult:          {result_text}")
    
    if pred == 1:
        print(f"\nAbnormal Score:  {prob:.1%}")
        print(f"Normal Score:    {1-prob:.1%}")
    else:
        print(f"\nNormal Score:    {1-prob:.1%}")
        print(f"Abnormal Score:  {prob:.1%}")
    
    print("\nRecommendation:")
    if prob > 0.7:
        print("  → High probability of abnormality detected")
        print("  → Recommend priority radiologist review")
    elif prob < 0.3:
        print("  → High probability of normal case")
        print("  → Can proceed to standard review queue")
    else:
        print("  → Borderline case with moderate confidence")
        print("  → Recommend careful radiologist examination")
    
    print("="*70)
    
    return pred, prob, true_label

def main():
    print("="*70)
    print("🔮 MULTI-PLANE MODEL - CASE PREDICTOR")
    print("="*70)
    
    CONFIG = {
        'mrnet_path': '/Users/yuvrajpratapsingh/knee-mri-abnormality-detection/dataset/MRNet-v1.0',
        'model_path': '/Users/yuvrajpratapsingh/knee-mri-abnormality-detection/outputs/models/best_model_multiplane.pth',
        'task': 'abnormal',
    }
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Load dataset
    print("\n📦 Loading validation dataset...")
    val_dataset = MultiPlaneMRNetDataset(
        root_dir=CONFIG['mrnet_path'],
        task=CONFIG['task'],
        split='valid',
        use_all_slices=False
    )
    
    # Load model
    print("🤖 Loading model...")
    model = MultiPlaneFusion(num_classes=1, dropout_rate=0.4)
    checkpoint = torch.load(CONFIG['model_path'], map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"✅ Model loaded (Epoch {checkpoint['epoch']+1})")
    print(f"   Sensitivity: {checkpoint['val_sensitivity']:.1%}")
    print(f"   Specificity: {checkpoint['val_specificity']:.1%}")
    
    # Available cases
    available_cases = val_dataset.labels_df['case'].values
    print(f"\n📊 Available cases: {available_cases.min()} to {available_cases.max()}")
    print(f"   Total: {len(available_cases)} cases")
    
    # Predict specific cases
    print("\n" + "="*70)
    print("🎯 EXAMPLE PREDICTIONS")
    print("="*70)
    
    # Show a few example predictions
    example_cases = [1130, 1150, 1200, 1230]  # Validation set case IDs
    
    for case_id in example_cases:
        if case_id in available_cases:
            try:
                predict_single_case(model, val_dataset, case_id, device)
                input("\nPress Enter to see next case...")
            except Exception as e:
                print(f"Error predicting case {case_id}: {e}")
        else:
            print(f"Case {case_id} not found in validation set")

if __name__ == '__main__':
    main()
