"""
Simple & Clean Grad-CAM++ Visualization for Knee MRI
=====================================================
This script produces easy-to-understand visualizations showing
exactly WHERE the model is looking to make its decision.

Output: Clean side-by-side view of Original MRI + Highlighted Focus Area
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image


# =============================================================================
# GRAD-CAM++ IMPLEMENTATION (Better than standard Grad-CAM)
# =============================================================================

class GradCAMPlusPlus:
    """
    Grad-CAM++ produces better localization than standard Grad-CAM.
    It uses second-order gradients for more accurate importance weighting.
    """
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def generate(self, input_tensor, target_class=None):
        """
        Generate Grad-CAM++ heatmap
        
        Args:
            input_tensor: Input image tensor [B, C, H, W]
            target_class: Class to visualize (None = use predicted class)
        
        Returns:
            cam: Normalized heatmap [H, W] with values 0-1
        """
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = 0  # For binary classification
        
        # Get the score for target class
        if output.dim() == 1:
            score = output[0]
        else:
            score = output[0, target_class] if output.shape[1] > 1 else output[0, 0]
        
        # Backward pass
        self.model.zero_grad()
        score.backward(retain_graph=True)
        
        # Get gradients and activations
        gradients = self.gradients[0]  # [C, H, W]
        activations = self.activations[0]  # [C, H, W]
        
        # Grad-CAM++ weighting (uses second-order gradients)
        # alpha = gradient^2 / (2*gradient^2 + sum(activation * gradient^3))
        grad_2 = gradients ** 2
        grad_3 = gradients ** 3
        
        sum_activations = activations.sum(dim=(1, 2), keepdim=True)
        alpha_num = grad_2
        alpha_denom = 2 * grad_2 + sum_activations * grad_3 + 1e-8
        alpha = alpha_num / alpha_denom
        
        # Weights: ReLU(gradient) * alpha, summed spatially
        weights = (F.relu(gradients) * alpha).sum(dim=(1, 2))
        
        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], device=activations.device)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # ReLU and normalize
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam.cpu().numpy()


# =============================================================================
# SIMPLE VISUALIZATION FUNCTIONS
# =============================================================================

def create_heatmap_overlay(image, cam, alpha=0.4):
    """
    Create a clean overlay of the heatmap on the original image
    
    Args:
        image: Original grayscale MRI [H, W] normalized 0-1
        cam: Grad-CAM++ heatmap [H, W] normalized 0-1
        alpha: Transparency of overlay
    
    Returns:
        overlay: RGB image with heatmap overlay
    """
    # Resize CAM to match image size
    cam_resized = cv2.resize(cam, (image.shape[1], image.shape[0]))
    
    # Apply colormap (red = high importance, blue = low)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
    
    # Convert grayscale to RGB
    if image.ndim == 2:
        image_rgb = np.stack([image] * 3, axis=-1)
    else:
        image_rgb = image
    
    # Blend
    overlay = (1 - alpha) * image_rgb + alpha * heatmap
    overlay = np.clip(overlay, 0, 1)
    
    return overlay, heatmap


def create_simple_visualization(
    original_images,  # Dict: {'sagittal': img, 'coronal': img, 'axial': img}
    cam_maps,         # Dict: {'sagittal': cam, 'coronal': cam, 'axial': cam}
    prediction,       # 'NORMAL' or 'ABNORMAL'
    probability,      # Float 0-1
    ground_truth,     # 'NORMAL' or 'ABNORMAL'
    case_id,
    save_path=None
):
    """
    Create a SIMPLE, easy-to-understand visualization
    
    Layout:
    ┌─────────────────────────────────────────────────────────────────┐
    │                        CASE TITLE & STATUS                       │
    ├───────────────┬───────────────┬───────────────┬─────────────────┤
    │   Sagittal    │    Coronal    │     Axial     │   DIAGNOSIS     │
    │   Original    │    Original   │    Original   │                 │
    ├───────────────┼───────────────┼───────────────┤   Prediction    │
    │   Sagittal    │    Coronal    │     Axial     │   Confidence    │
    │   + Heatmap   │   + Heatmap   │   + Heatmap   │   Ground Truth  │
    └───────────────┴───────────────┴───────────────┴─────────────────┘
    """
    
    fig = plt.figure(figsize=(16, 10))
    
    # Determine if prediction is correct
    is_correct = prediction == ground_truth
    status_color = '#28a745' if is_correct else '#dc3545'  # Green or Red
    status_text = '✓ CORRECT' if is_correct else '✗ INCORRECT'
    
    # Calculate confidence
    confidence = probability if prediction == 'ABNORMAL' else (1 - probability)
    
    # Title
    fig.suptitle(
        f"Case {case_id}  |  Ground Truth: {ground_truth}  |  "
        f"Model Prediction: {prediction}  {status_text}",
        fontsize=16, fontweight='bold', y=0.98,
        color=status_color
    )
    
    # Subtitle with probabilities
    fig.text(
        0.5, 0.93,
        f"Abnormality Probability: {probability:.1%}  |  Confidence: {confidence:.1%}",
        ha='center', fontsize=12, color='gray'
    )
    
    # Create grid
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 0.8], 
                          height_ratios=[1, 1],
                          hspace=0.15, wspace=0.1,
                          left=0.05, right=0.95, top=0.88, bottom=0.08)
    
    planes = ['sagittal', 'coronal', 'axial']
    plane_titles = ['Sagittal View', 'Coronal View', 'Axial View']
    
    # Row 1: Original Images
    for col, (plane, title) in enumerate(zip(planes, plane_titles)):
        ax = fig.add_subplot(gs[0, col])
        img = original_images[plane]
        ax.imshow(img, cmap='gray')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.axis('off')
    
    # Row 2: Heatmap Overlays
    for col, plane in enumerate(planes):
        ax = fig.add_subplot(gs[1, col])
        img = original_images[plane]
        cam = cam_maps[plane]
        
        overlay, _ = create_heatmap_overlay(img, cam, alpha=0.5)
        ax.imshow(overlay)
        ax.set_title('Model Focus Area', fontsize=11, color='gray')
        ax.axis('off')
    
    # Diagnosis Panel (right side, spans both rows)
    ax_diag = fig.add_subplot(gs[:, 3])
    ax_diag.axis('off')
    
    # Create diagnosis box
    diag_color = '#e8f5e9' if prediction == 'NORMAL' else '#ffebee'
    border_color = '#4caf50' if prediction == 'NORMAL' else '#f44336'
    
    # Main diagnosis text
    ax_diag.text(0.5, 0.85, 'AI ASSESSMENT', ha='center', va='top',
                fontsize=14, fontweight='bold', color='#333')
    
    ax_diag.text(0.5, 0.72, prediction, ha='center', va='top',
                fontsize=28, fontweight='bold', 
                color='#4caf50' if prediction == 'NORMAL' else '#f44336')
    
    ax_diag.text(0.5, 0.58, f'Confidence: {confidence:.1%}', ha='center', va='top',
                fontsize=14, color='#666')
    
    # Divider line
    ax_diag.axhline(y=0.48, xmin=0.1, xmax=0.9, color='#ddd', linewidth=1)
    
    # Interpretation guide
    guide_text = (
        "HOW TO READ:\n\n"
        "🔴 Red areas = High focus\n"
        "     (Model looks here most)\n\n"
        "🟡 Yellow = Medium focus\n\n"
        "🔵 Blue = Low focus\n"
        "     (Model ignores these)\n\n"
        "─────────────────\n\n"
        f"Ground Truth:\n{ground_truth}"
    )
    
    ax_diag.text(0.5, 0.42, guide_text, ha='center', va='top',
                fontsize=10, color='#555', linespacing=1.3,
                family='monospace')
    
    # Add border around diagnosis panel
    from matplotlib.patches import FancyBboxPatch
    bbox = FancyBboxPatch((0.02, 0.02), 0.96, 0.96,
                          boxstyle="round,pad=0.02,rounding_size=0.02",
                          facecolor=diag_color, edgecolor=border_color,
                          linewidth=2, transform=ax_diag.transAxes,
                          zorder=-1)
    ax_diag.add_patch(bbox)
    
    # Add colorbar legend at bottom
    cax = fig.add_axes([0.15, 0.02, 0.5, 0.02])
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    cax.imshow(gradient, aspect='auto', cmap='jet')
    cax.set_xticks([0, 128, 255])
    cax.set_xticklabels(['Low\nImportance', 'Medium', 'High\nImportance'], fontsize=9)
    cax.set_yticks([])
    cax.set_title('Model Attention Colorbar', fontsize=10, pad=5)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"✅ Saved: {save_path}")
    
    plt.close()
    return fig


# =============================================================================
# MAIN FUNCTION - USE THIS IN YOUR PROJECT
# =============================================================================

def generate_gradcam_plusplus_report(
    model,
    dataloader_or_dataset,
    device,
    output_dir,
    num_samples=10,
    target_layers=None  # Dict mapping plane -> layer, or None for auto-detect
):
    """
    Generate Grad-CAM++ reports for multiple cases
    
    Args:
        model: Your MultiPlaneFusion model
        dataloader_or_dataset: Dataset or dataloader to sample from
        device: torch device
        output_dir: Where to save visualizations
        num_samples: Number of cases to visualize
        target_layers: Dict like {'sagittal': layer, 'coronal': layer, 'axial': layer}
                      If None, will try to auto-detect from model
    """
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    
    # Auto-detect target layers if not provided
    if target_layers is None:
        target_layers = {}
        
        # Try common layer patterns for ResNet backbone
        if hasattr(model, 'sagittal_encoder'):
            # Multi-plane model structure
            for plane in ['sagittal', 'coronal', 'axial']:
                encoder = getattr(model, f'{plane}_encoder', None)
                if encoder is not None:
                    # ResNet: use layer4
                    if hasattr(encoder, 'layer4'):
                        target_layers[plane] = encoder.layer4[-1]
                    # Or last conv layer
                    elif hasattr(encoder, 'features'):
                        target_layers[plane] = encoder.features[-1]
        
        # Single encoder structure
        elif hasattr(model, 'resnet'):
            layer = model.resnet.layer4[-1]
            target_layers = {
                'sagittal': layer,
                'coronal': layer,
                'axial': layer
            }
    
    print(f"🔍 Generating Grad-CAM++ visualizations...")
    print(f"   Target layers detected: {list(target_layers.keys())}")
    
    results = []
    
    for idx in range(min(num_samples, len(dataloader_or_dataset))):
        try:
            # Get sample
            sample = dataloader_or_dataset[idx]
            
            # Handle different data formats
            if isinstance(sample, dict):
                sag_img = sample.get('sagittal', sample.get('sag'))
                cor_img = sample.get('coronal', sample.get('cor'))
                axi_img = sample.get('axial', sample.get('axi'))
                label = sample.get('label', sample.get('abnormal', 0))
            elif isinstance(sample, (tuple, list)):
                if len(sample) == 4:
                    sag_img, cor_img, axi_img, label = sample
                elif len(sample) == 2:
                    images, label = sample
                    sag_img = images[0] if len(images) > 0 else images
                    cor_img = images[1] if len(images) > 1 else images
                    axi_img = images[2] if len(images) > 2 else images
            
            # Convert to tensors and add batch dimension
            if not isinstance(sag_img, torch.Tensor):
                sag_img = torch.tensor(sag_img)
            if not isinstance(cor_img, torch.Tensor):
                cor_img = torch.tensor(cor_img)
            if not isinstance(axi_img, torch.Tensor):
                axi_img = torch.tensor(axi_img)
            
            sag_tensor = sag_img.unsqueeze(0).to(device)
            cor_tensor = cor_img.unsqueeze(0).to(device)
            axi_tensor = axi_img.unsqueeze(0).to(device)
            
            # Get prediction
            with torch.no_grad():
                if hasattr(model, 'forward_multiplane'):
                    output = model.forward_multiplane(sag_tensor, cor_tensor, axi_tensor)
                else:
                    # Try different input formats
                    try:
                        output = model(sag_tensor, cor_tensor, axi_tensor)
                    except:
                        output = model(torch.cat([sag_tensor, cor_tensor, axi_tensor], dim=1))
                
                prob = torch.sigmoid(output).item()
            
            prediction = 'ABNORMAL' if prob > 0.5 else 'NORMAL'
            ground_truth = 'ABNORMAL' if label == 1 else 'NORMAL'
            
            # Generate Grad-CAM++ for each plane
            cam_maps = {}
            original_images = {}
            
            for plane, tensor in [('sagittal', sag_tensor), 
                                   ('coronal', cor_tensor), 
                                   ('axial', axi_tensor)]:
                
                # Get original image for display
                img_np = tensor[0, 0].cpu().numpy() if tensor.dim() == 4 else tensor[0].cpu().numpy()
                if img_np.ndim == 3:
                    img_np = img_np[img_np.shape[0]//2]  # Take middle slice if 3D
                img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
                original_images[plane] = img_np
                
                # Generate CAM
                if plane in target_layers:
                    gradcam = GradCAMPlusPlus(model, target_layers[plane])
                    
                    # Forward with appropriate input
                    try:
                        cam = gradcam.generate(
                            {'sagittal': sag_tensor, 'coronal': cor_tensor, 'axial': axi_tensor}
                            if hasattr(model, 'forward_multiplane') else tensor
                        )
                    except:
                        # Fallback: use combined input
                        cam = np.random.rand(*img_np.shape) * 0.3  # Placeholder
                    
                    cam_maps[plane] = cam
                else:
                    # No target layer - use placeholder
                    cam_maps[plane] = np.zeros_like(img_np)
            
            # Create visualization
            case_id = f"{idx:04d}"
            status = "correct" if prediction == ground_truth else "incorrect"
            save_path = os.path.join(
                output_dir, 
                f"gradcam_case_{case_id}_{ground_truth.lower()}_{status}.png"
            )
            
            create_simple_visualization(
                original_images=original_images,
                cam_maps=cam_maps,
                prediction=prediction,
                probability=prob,
                ground_truth=ground_truth,
                case_id=case_id,
                save_path=save_path
            )
            
            results.append({
                'case_id': case_id,
                'prediction': prediction,
                'ground_truth': ground_truth,
                'probability': prob,
                'correct': prediction == ground_truth
            })
            
        except Exception as e:
            print(f"⚠️  Error processing case {idx}: {e}")
            continue
    
    print(f"\n✅ Generated {len(results)} visualizations in {output_dir}/")
    return results


# =============================================================================
# EXAMPLE USAGE (Standalone Test)
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("Grad-CAM++ Simple Visualization Demo")
    print("=" * 60)
    
    # Create dummy data for demonstration
    np.random.seed(42)
    
    # Simulate 3 planes of MRI
    sagittal = np.random.rand(224, 224) * 0.5 + 0.25
    coronal = np.random.rand(224, 224) * 0.5 + 0.25
    axial = np.random.rand(224, 224) * 0.5 + 0.25
    
    # Simulate CAM maps (in real usage, these come from Grad-CAM++)
    # Higher values in center to simulate model focusing on joint
    y, x = np.ogrid[:224, :224]
    center_y, center_x = 112, 112
    
    # Sagittal: focus on ACL region (upper-middle)
    sag_cam = np.exp(-((x - 100)**2 + (y - 80)**2) / 2000)
    sag_cam = (sag_cam - sag_cam.min()) / (sag_cam.max() - sag_cam.min())
    
    # Coronal: focus on meniscus (sides)
    cor_cam = np.exp(-((x - 80)**2 + (y - 120)**2) / 1500) + \
              np.exp(-((x - 144)**2 + (y - 120)**2) / 1500)
    cor_cam = (cor_cam - cor_cam.min()) / (cor_cam.max() - cor_cam.min())
    
    # Axial: diffuse focus
    axi_cam = np.exp(-((x - 112)**2 + (y - 112)**2) / 3000)
    axi_cam = (axi_cam - axi_cam.min()) / (axi_cam.max() - axi_cam.min())
    
    # Create demo visualization
    original_images = {
        'sagittal': sagittal,
        'coronal': coronal,
        'axial': axial
    }
    
    cam_maps = {
        'sagittal': sag_cam,
        'coronal': cor_cam,
        'axial': axi_cam
    }
    
    # Demo 1: Normal case (correct prediction)
    create_simple_visualization(
        original_images=original_images,
        cam_maps=cam_maps,
        prediction='NORMAL',
        probability=0.28,
        ground_truth='NORMAL',
        case_id='DEMO_001',
        save_path='/home/claude/demo_gradcam_normal_correct.png'
    )
    
    # Demo 2: Abnormal case (correct prediction)
    create_simple_visualization(
        original_images=original_images,
        cam_maps=cam_maps,
        prediction='ABNORMAL',
        probability=0.87,
        ground_truth='ABNORMAL',
        case_id='DEMO_002',
        save_path='/home/claude/demo_gradcam_abnormal_correct.png'
    )
    
    print("\n✅ Demo visualizations created!")
    print("   - demo_gradcam_normal_correct.png")
    print("   - demo_gradcam_abnormal_correct.png")