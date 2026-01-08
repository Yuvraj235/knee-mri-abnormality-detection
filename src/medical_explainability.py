"""
Grad-CAM Explanations for Medical Interpretability
Shows doctors WHERE the model is looking
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Dict, Tuple
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.medical_uncertainty_model import MedicalUncertaintyModel


class MedicalExplainer:
    """
    Generate visual explanations for medical professionals
    """
    
    def __init__(self, model: MedicalUncertaintyModel):
        self.model = model
        self.gradients = {}
        self.activations = {}
        
        # Register hooks for Grad-CAM
        self._register_hooks()
    
    def _register_hooks(self):
        """Register hooks to capture gradients and activations"""
        
        def get_forward_hook(name):
            def forward_hook(module, input, output):
                self.activations[name] = output.detach()
            return forward_hook
        
        def get_backward_hook(name):
            def backward_hook(module, grad_input, grad_output):
                self.gradients[name] = grad_output[0].detach()
            return backward_hook
        
        # Find the layer4 (last conv block) for each encoder
        encoders = {
            'sagittal': self.model.model.sagittal_encoder,
            'coronal': self.model.model.coronal_encoder,
            'axial': self.model.model.axial_encoder
        }
        
        for encoder_name, encoder in encoders.items():
            if len(list(encoder.children())) >= 2:
                target_layer = list(encoder.children())[-2]
                target_layer.register_forward_hook(get_forward_hook(encoder_name))
                target_layer.register_full_backward_hook(get_backward_hook(encoder_name))
    
    def generate_gradcam(
        self,
        sagittal: torch.Tensor,
        coronal: torch.Tensor,
        axial: torch.Tensor,
        target_plane: str = 'sagittal'
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap for a specific plane
        """
        
        self.model.model.eval()
        
        # Move to device and require gradients for target plane
        sagittal = sagittal.to(self.model.device)
        coronal = coronal.to(self.model.device)
        axial = axial.to(self.model.device)
        
        if target_plane == 'sagittal':
            sagittal = sagittal.requires_grad_(True)
        elif target_plane == 'coronal':
            coronal = coronal.requires_grad_(True)
        elif target_plane == 'axial':
            axial = axial.requires_grad_(True)
        
        # Forward pass
        logits = self.model.model(sagittal, coronal, axial)
        
        # Backward pass
        self.model.model.zero_grad()
        logits.backward()
        
        # Get gradients and activations for target plane
        if target_plane not in self.activations or target_plane not in self.gradients:
            print(f"⚠️  No activations captured for {target_plane}, returning zero heatmap")
            return np.zeros((224, 224))
        
        gradients = self.gradients[target_plane]
        activations = self.activations[target_plane]
        
        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        
        # Weighted combination of activations
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        
        # Resize to input size
        cam = F.interpolate(
            cam,
            size=(224, 224),
            mode='bilinear',
            align_corners=False
        )
        
        # Normalize
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        else:
            cam = np.zeros_like(cam)
        
        return cam
    
    def generate_all_gradcams(
        self,
        sagittal: torch.Tensor,
        coronal: torch.Tensor,
        axial: torch.Tensor
    ) -> Dict[str, np.ndarray]:
        """
        Generate Grad-CAM heatmaps for all three planes
        """
        
        heatmaps = {}
        
        for plane in ['sagittal', 'coronal', 'axial']:
            try:
                heatmap = self.generate_gradcam(sagittal, coronal, axial, target_plane=plane)
                # Ensure it's a valid numpy array
                if not isinstance(heatmap, np.ndarray):
                    heatmap = np.zeros((224, 224))
                heatmaps[plane] = heatmap
            except Exception as e:
                print(f"⚠️  Error generating Grad-CAM for {plane}: {e}")
                heatmaps[plane] = np.zeros((224, 224))
        
        return heatmaps
    
    def visualize_explanation(
        self,
        images: Dict[str, np.ndarray],
        heatmaps: Dict[str, np.ndarray],
        prediction: Dict[str, any],
        save_path: str = None
    ):
        """
        Create visualization for doctors
        """
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        planes = ['sagittal', 'coronal', 'axial']
        
        for idx, plane in enumerate(planes):
            # Ensure we have valid data
            img = images.get(plane, np.zeros((224, 224)))
            heatmap = heatmaps.get(plane, np.zeros((224, 224)))
            
            # Convert to numpy if needed
            if not isinstance(img, np.ndarray):
                img = np.array(img)
            if not isinstance(heatmap, np.ndarray):
                heatmap = np.array(heatmap)
            
            # Ensure 2D
            if len(img.shape) > 2:
                img = img[:, :, 0] if img.shape[2] > 0 else img.squeeze()
            if len(heatmap.shape) > 2:
                heatmap = heatmap[:, :, 0] if heatmap.shape[2] > 0 else heatmap.squeeze()
            
            # Original image
            ax_img = axes[0, idx]
            ax_img.imshow(img, cmap='gray')
            ax_img.set_title(f'{plane.capitalize()} View', fontsize=14, fontweight='bold')
            ax_img.axis('off')
            
            # Heatmap overlay
            ax_heat = axes[1, idx]
            ax_heat.imshow(img, cmap='gray')
            ax_heat.imshow(heatmap, cmap='jet', alpha=0.5)
            ax_heat.set_title(f'Model Attention - {plane.capitalize()}', fontsize=14, fontweight='bold')
            ax_heat.axis('off')
        
        # Add prediction info
        pred_text = f"""
PREDICTION: {prediction['prediction_label']}
Probability: {prediction['probability']:.1%}
Confidence: {prediction['confidence']:.1%}
Uncertainty: {prediction.get('uncertainty', 0):.3f}
        """
        
        fig.text(0.5, 0.96, pred_text, 
                ha='center', fontsize=16, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout(rect=[0, 0, 1, 0.94])
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Saved explanation to: {save_path}")
        
        plt.close()


def test_explainability():
    """Test the explainability module"""
    
    print("\n" + "="*70)
    print("🔬 TESTING MEDICAL EXPLAINABILITY")
    print("="*70)
    
    from src.multiplane_loader import MultiPlaneMRNetDataset
    
    model_path = '/Users/yuvrajpratapsingh/knee-mri-abnormality-detection/outputs/resnet_only/best_model.pth'
    dataset_path = '/Users/yuvrajpratapsingh/knee-mri-abnormality-detection/dataset/MRNet-v1.0'
    
    # Load model
    print("\n📦 Loading model...")
    medical_model = MedicalUncertaintyModel(model_path)
    
    print("\n🔧 Setting up explainer...")
    explainer = MedicalExplainer(medical_model)
    
    # Load a real MRI case
    print("\n📊 Loading test case...")
    dataset = MultiPlaneMRNetDataset(dataset_path, task='abnormal', split='valid', use_all_slices=False)
    sample = dataset[0]
    
    # Get prediction with uncertainty
    print("\n🔍 Generating prediction with uncertainty...")
    prediction = medical_model.predict_with_uncertainty(
        sample['sagittal'].unsqueeze(0),
        sample['coronal'].unsqueeze(0),
        sample['axial'].unsqueeze(0)
    )
    
    print(f"\n✅ Prediction: {prediction['prediction_label']}")
    print(f"   Probability: {prediction['probability']:.3f}")
    print(f"   Confidence: {prediction['confidence']:.3f}")
    
    # Generate explanations
    print("\n🎨 Generating Grad-CAM explanations...")
    heatmaps = explainer.generate_all_gradcams(
        sample['sagittal'].unsqueeze(0),
        sample['coronal'].unsqueeze(0),
        sample['axial'].unsqueeze(0)
    )
    
    print("✅ Grad-CAM heatmaps generated")
    
    # Convert tensors to numpy for visualization
    images = {
        'sagittal': sample['sagittal'].permute(1, 2, 0).numpy()[:, :, 0],
        'coronal': sample['coronal'].permute(1, 2, 0).numpy()[:, :, 0],
        'axial': sample['axial'].permute(1, 2, 0).numpy()[:, :, 0]
    }
    
    # Visualize
    print("\n📊 Creating visualization...")
    explainer.visualize_explanation(
        images,
        heatmaps,
        prediction,
        save_path='outputs/medical_explanation_example.png'
    )
    
    print("\n" + "="*70)
    print("✅ EXPLAINABILITY MODULE READY!")
    print("="*70)


if __name__ == '__main__':
    test_explainability()
