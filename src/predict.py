import os
import sys
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import ResNetDeiTFusion
from src.data_loader import MRNetDataset

def predict_single_image(model, image_tensor, device):
    """Make prediction on a single image"""
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.unsqueeze(0).to(device)  # Add batch dimension
        output = model(image_tensor)
        probability = torch.sigmoid(output).item()
        prediction = 1 if probability > 0.5 else 0
    
    return prediction, probability

def visualize_prediction(image_tensor, prediction, probability, save_path=None):
    """Visualize image with prediction"""
    # Convert tensor to numpy for visualization
    img_np = image_tensor.cpu().permute(1, 2, 0).numpy()
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
    
    plt.figure(figsize=(8, 6))
    plt.imshow(img_np)
    plt.axis('off')
    
    label = "ABNORMAL" if prediction == 1 else "NORMAL"
    color = "red" if prediction == 1 else "green"
    plt.title(f"Prediction: {label}\nConfidence: {probability:.2%}", 
              fontsize=16, color=color, weight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Visualization saved to {save_path}")
    
    plt.show()

def main():
    # Configuration
    MODEL_PATH = '/Users/yuvrajpratapsingh/knee-mri-abnormality-detection/outputs/models/best_model.pth'
    DATASET_PATH = '/Users/yuvrajpratapsingh/knee-mri-abnormality-detection/dataset/MRNet-v1.0'
    OUTPUT_DIR = '/Users/yuvrajpratapsingh/knee-mri-abnormality-detection/outputs/predictions'
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🖥️  Using device: {device}")
    
    # Load model
    print("\n📦 Loading model...")
    model = ResNetDeiTFusion(num_classes=1, fusion_type='attention', pretrained=False)
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    print(f"✅ Model loaded (Validation AUC: {checkpoint['val_auc']:.4f})")
    
    # Load validation dataset
    print("\n📊 Loading validation dataset...")
    val_dataset = MRNetDataset(
        root_dir=DATASET_PATH,
        plane='sagittal',
        task='abnormal',
        split='valid',
        use_all_slices=False
    )
    
    # Make predictions on first 10 samples
    print("\n🔍 Making predictions...")
    for i in range(min(10, len(val_dataset))):
        image, true_label = val_dataset[i]
        prediction, probability = predict_single_image(model, image, device)
        
        print(f"\nSample {i+1}:")
        print(f"   True label: {'ABNORMAL' if true_label == 1 else 'NORMAL'}")
        print(f"   Prediction: {'ABNORMAL' if prediction == 1 else 'NORMAL'}")
        print(f"   Confidence: {probability:.2%}")
        print(f"   Correct: {'✅' if prediction == true_label else '❌'}")
        
        # Visualize
        save_path = f"{OUTPUT_DIR}/prediction_{i+1}.png"
        visualize_prediction(image, prediction, probability, save_path)

if __name__ == '__main__':
    main()
