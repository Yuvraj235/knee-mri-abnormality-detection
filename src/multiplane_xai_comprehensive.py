import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from scipy.ndimage import gaussian_filter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.multiplane_loader import MultiPlaneMRNetDataset
from src.multiplane_model import MultiPlaneFusion

# ============================================================================
# METHOD 1: OCCLUSION SENSITIVITY (Most Reliable!)
# ============================================================================

class OcclusionSensitivity:
    """
    Occlusion Sensitivity - The most reliable XAI method
    Systematically occludes parts of the image and measures impact on prediction
    
    Pros: Model-agnostic, very interpretable, always works
    Cons: Slow (but worth it!)
    """
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
    
    def generate(self, sagittal, coronal, axial, patch_size=16, stride=8):
        """
        Generate occlusion sensitivity map
        
        Args:
            sagittal, coronal, axial: Input tensors
            patch_size: Size of occlusion patch
            stride: Step size for sliding window
        """
        self.model.eval()
        
        # Get baseline prediction
        with torch.no_grad():
            baseline_output = self.model(sagittal, coronal, axial)
            baseline_prob = torch.sigmoid(baseline_output).item()
        
        # Generate sensitivity maps for each plane
        maps = {}
        
        for plane_name, plane_tensor in [('sagittal', sagittal), 
                                          ('coronal', coronal), 
                                          ('axial', axial)]:
            
            _, _, h, w = plane_tensor.shape
            sensitivity_map = np.zeros((h, w))
            
            # Slide window across image
            for i in range(0, h - patch_size + 1, stride):
                for j in range(0, w - patch_size + 1, stride):
                    # Create occluded version
                    occluded = plane_tensor.clone()
                    occluded[:, :, i:i+patch_size, j:j+patch_size] = 0
                    
                    # Get prediction with occlusion
                    with torch.no_grad():
                        if plane_name == 'sagittal':
                            output = self.model(occluded, coronal, axial)
                        elif plane_name == 'coronal':
                            output = self.model(sagittal, occluded, axial)
                        else:
                            output = self.model(sagittal, coronal, occluded)
                        
                        occluded_prob = torch.sigmoid(output).item()
                    
                    # Measure change in prediction
                    importance = abs(baseline_prob - occluded_prob)
                    sensitivity_map[i:i+patch_size, j:j+patch_size] = importance
            
            # Normalize
            if sensitivity_map.max() > 0:
                sensitivity_map = sensitivity_map / sensitivity_map.max()
            
            maps[plane_name] = sensitivity_map
        
        return maps

# ============================================================================
# METHOD 2: INTEGRATED GRADIENTS
# ============================================================================

class IntegratedGradients:
    """
    Integrated Gradients - Theoretically sound attribution method
    
    Pros: Theoretically grounded, satisfies axioms
    Cons: Requires multiple forward passes
    """
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
    
    def generate(self, sagittal, coronal, axial, steps=50):
        """Generate integrated gradients attribution"""
        
        self.model.eval()
        
        # Baseline (black image)
        baseline_sag = torch.zeros_like(sagittal)
        baseline_cor = torch.zeros_like(coronal)
        baseline_axi = torch.zeros_like(axial)
        
        # Store gradients
        integrated_grads = {
            'sagittal': torch.zeros_like(sagittal),
            'coronal': torch.zeros_like(coronal),
            'axial': torch.zeros_like(axial)
        }
        
        # Interpolate between baseline and input
        for alpha in np.linspace(0, 1, steps):
            # Interpolated inputs
            interp_sag = baseline_sag + alpha * (sagittal - baseline_sag)
            interp_cor = baseline_cor + alpha * (coronal - baseline_cor)
            interp_axi = baseline_axi + alpha * (axial - baseline_axi)
            
            # Require gradients
            interp_sag.requires_grad = True
            interp_cor.requires_grad = True
            interp_axi.requires_grad = True
            
            # Forward pass
            output = self.model(interp_sag, interp_cor, interp_axi)
            
            # Backward pass
            self.model.zero_grad()
            output.backward()
            
            # Accumulate gradients
            integrated_grads['sagittal'] += interp_sag.grad.detach()
            integrated_grads['coronal'] += interp_cor.grad.detach()
            integrated_grads['axial'] += interp_axi.grad.detach()
        
        # Average and multiply by input difference
        for key in integrated_grads.keys():
            integrated_grads[key] /= steps
        
        integrated_grads['sagittal'] *= (sagittal - baseline_sag)
        integrated_grads['coronal'] *= (coronal - baseline_cor)
        integrated_grads['axial'] *= (axial - baseline_axi)
        
        # Convert to numpy and take max across channels
        maps = {}
        for key, grad in integrated_grads.items():
            attr_map = grad.abs().max(dim=1)[0].squeeze().cpu().numpy()
            # Normalize
            if attr_map.max() > 0:
                attr_map = (attr_map - attr_map.min()) / (attr_map.max() - attr_map.min())
            maps[key] = attr_map
        
        return maps

# ============================================================================
# METHOD 3: SMOOTH GRADIENTS
# ============================================================================

class SmoothGradients:
    """
    Smooth Gradients - Reduces noise by averaging over noisy inputs
    
    Pros: Less noisy than vanilla gradients
    Cons: Still gradient-based
    """
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
    
    def generate(self, sagittal, coronal, axial, n_samples=50, noise_level=0.1):
        """Generate smooth gradients"""
        
        self.model.eval()
        
        # Accumulate gradients
        smooth_grads = {
            'sagittal': torch.zeros_like(sagittal),
            'coronal': torch.zeros_like(coronal),
            'axial': torch.zeros_like(axial)
        }
        
        for _ in range(n_samples):
            # Add noise
            noisy_sag = sagittal + torch.randn_like(sagittal) * noise_level
            noisy_cor = coronal + torch.randn_like(coronal) * noise_level
            noisy_axi = axial + torch.randn_like(axial) * noise_level
            
            # Require gradients
            noisy_sag.requires_grad = True
            noisy_cor.requires_grad = True
            noisy_axi.requires_grad = True
            
            # Forward pass
            output = self.model(noisy_sag, noisy_cor, noisy_axi)
            
            # Backward pass
            self.model.zero_grad()
            output.backward()
            
            # Accumulate
            smooth_grads['sagittal'] += noisy_sag.grad.detach()
            smooth_grads['coronal'] += noisy_cor.grad.detach()
            smooth_grads['axial'] += noisy_axi.grad.detach()
        
        # Average
        for key in smooth_grads.keys():
            smooth_grads[key] /= n_samples
        
        # Convert to numpy
        maps = {}
        for key, grad in smooth_grads.items():
            grad_map = grad.abs().max(dim=1)[0].squeeze().cpu().numpy()
            # Normalize
            if grad_map.max() > 0:
                grad_map = (grad_map - grad_map.min()) / (grad_map.max() - grad_map.min())
            maps[key] = grad_map
        
        return maps

# ============================================================================
# METHOD 4: INPUT × GRADIENT (FIXED!)
# ============================================================================

class InputTimesGradient:
    """
    Input × Gradient - Simple but effective
    
    Pros: Fast, simple, often works well
    Cons: Can be noisy
    """
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
    
    def generate(self, sagittal, coronal, axial):
        """Generate input × gradient attribution"""
        
        self.model.eval()
        
        # Require gradients
        sagittal.requires_grad = True
        coronal.requires_grad = True
        axial.requires_grad = True
        
        # Forward pass
        output = self.model(sagittal, coronal, axial)
        
        # Backward pass
        self.model.zero_grad()
        output.backward()
        
        # Multiply input by gradient - FIXED: Added .detach()
        sag_attr = (sagittal * sagittal.grad).abs().max(dim=1)[0].squeeze().detach().cpu().numpy()
        cor_attr = (coronal * coronal.grad).abs().max(dim=1)[0].squeeze().detach().cpu().numpy()
        axi_attr = (axial * axial.grad).abs().max(dim=1)[0].squeeze().detach().cpu().numpy()
        
        # Normalize
        maps = {}
        for name, attr in [('sagittal', sag_attr), ('coronal', cor_attr), ('axial', axi_attr)]:
            if attr.max() > 0:
                attr = (attr - attr.min()) / (attr.max() - attr.min())
            maps[name] = attr
        
        return maps

# ============================================================================
# METHOD 5: SIMPLE GRADIENT (Baseline)
# ============================================================================

class SimpleGradient:
    """
    Simple Gradient - Basic gradient-based attribution
    
    Pros: Very fast
    Cons: Noisy, less reliable
    """
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
    
    def generate(self, sagittal, coronal, axial):
        """Generate simple gradient attribution"""
        
        self.model.eval()
        
        # Require gradients
        sagittal.requires_grad = True
        coronal.requires_grad = True
        axial.requires_grad = True
        
        # Forward pass
        output = self.model(sagittal, coronal, axial)
        
        # Backward pass
        self.model.zero_grad()
        output.backward()
        
        # Get gradients
        sag_grad = sagittal.grad.abs().max(dim=1)[0].squeeze().detach().cpu().numpy()
        cor_grad = coronal.grad.abs().max(dim=1)[0].squeeze().detach().cpu().numpy()
        axi_grad = axial.grad.abs().max(dim=1)[0].squeeze().detach().cpu().numpy()
        
        # Normalize
        maps = {}
        for name, grad in [('sagittal', sag_grad), ('coronal', cor_grad), ('axial', axi_grad)]:
            if grad.max() > 0:
                grad = (grad - grad.min()) / (grad.max() - grad.min())
            maps[name] = grad
        
        return maps

# ============================================================================
# VISUALIZATION
# ============================================================================

def apply_heatmap(img, attribution_map, alpha=0.4):
    """Apply colormap overlay"""
    # Resize to match image
    attr_resized = cv2.resize(attribution_map, (img.shape[1], img.shape[0]))
    
    # Apply colormap
    heatmap = cv2.applyColorMap(np.uint8(255 * attr_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Overlay
    overlaid = (1 - alpha) * img + alpha * heatmap
    return np.clip(overlaid, 0, 255).astype(np.uint8)

def analyze_case_comprehensive(model, dataset, case_idx, device, output_dir):
    """Analyze case with ALL XAI methods"""
    
    # Get case data
    batch = dataset[case_idx]
    case_id = dataset.labels_df.iloc[case_idx]['case']
    true_label = batch['label'].item()
    
    sagittal = batch['sagittal'].unsqueeze(0).to(device)
    coronal = batch['coronal'].unsqueeze(0).to(device)
    axial = batch['axial'].unsqueeze(0).to(device)
    
    # Get prediction
    model.eval()
    with torch.no_grad():
        output = model(sagittal, coronal, axial)
        prob = torch.sigmoid(output).item()
        pred = 1 if prob > 0.5 else 0
    
    # Denormalize images
    def denormalize(tensor):
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = tensor.cpu() * std + mean
        tensor = torch.clamp(tensor, 0, 1)
        return (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    
    sag_img = denormalize(batch['sagittal'])
    cor_img = denormalize(batch['coronal'])
    axi_img = denormalize(batch['axial'])
    
    print(f"   Analyzing case {case_id} with 5 XAI methods...")
    
    # Generate all attributions
    print(f"     → Occlusion Sensitivity (slow but reliable)...")
    occlusion = OcclusionSensitivity(model, device)
    occlusion_maps = occlusion.generate(sagittal.clone(), coronal.clone(), axial.clone(), 
                                        patch_size=28, stride=14)
    
    print(f"     → Integrated Gradients...")
    ig = IntegratedGradients(model, device)
    ig_maps = ig.generate(sagittal.clone(), coronal.clone(), axial.clone(), steps=30)
    
    print(f"     → Smooth Gradients...")
    smooth = SmoothGradients(model, device)
    smooth_maps = smooth.generate(sagittal.clone(), coronal.clone(), axial.clone(), n_samples=30)
    
    print(f"     → Input × Gradient...")
    input_grad = InputTimesGradient(model, device)
    input_grad_maps = input_grad.generate(sagittal.clone(), coronal.clone(), axial.clone())
    
    print(f"     → Simple Gradient...")
    simple = SimpleGradient(model, device)
    simple_maps = simple.generate(sagittal.clone(), coronal.clone(), axial.clone())
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(24, 18))
    gs = fig.add_gridspec(6, 4, hspace=0.4, wspace=0.3)
    
    # Title
    title = f"Case {case_id:04d} | True: {'Abnormal' if true_label == 1 else 'Normal'} | "
    title += f"Predicted: {'Abnormal' if pred == 1 else 'Normal'} ({prob:.1%})"
    if pred == true_label:
        title += " ✅"
        color = 'green'
    else:
        title += " ❌"
        color = 'red'
    
    fig.suptitle(title, fontsize=18, weight='bold', color=color)
    
    # Row 0: Original images
    images = [
        ('Sagittal', sag_img, 0),
        ('Coronal', cor_img, 1),
        ('Axial', axi_img, 2)
    ]
    
    for name, img, col in images:
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(img)
        ax.set_title(f'{name} View', fontsize=12, weight='bold')
        ax.axis('off')
    
    # Prediction info
    ax = fig.add_subplot(gs[0, 3])
    ax.axis('off')
    info_text = f"PREDICTION\n\n"
    info_text += f"Prob: {prob:.1%}\n"
    info_text += f"Conf: {abs(prob-0.5)*2:.1%}\n\n"
    if prob > 0.5:
        info_text += f"Abnormal: {prob:.1%}\n"
        info_text += f"Normal: {1-prob:.1%}\n"
    else:
        info_text += f"Normal: {1-prob:.1%}\n"
        info_text += f"Abnormal: {prob:.1%}"
    ax.text(0.1, 0.5, info_text, fontsize=11, verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7),
            family='monospace')
    
    # Rows 1-5: Different XAI methods
    methods = [
        ('Occlusion Sensitivity', occlusion_maps, 1),
        ('Integrated Gradients', ig_maps, 2),
        ('Smooth Gradients', smooth_maps, 3),
        ('Input × Gradient', input_grad_maps, 4),
        ('Simple Gradient', simple_maps, 5)
    ]
    
    for method_name, maps, row in methods:
        for plane_name, plane_img, col in [('sagittal', sag_img, 0), 
                                             ('coronal', cor_img, 1), 
                                             ('axial', axi_img, 2)]:
            ax = fig.add_subplot(gs[row, col])
            overlay = apply_heatmap(plane_img, maps[plane_name])
            ax.imshow(overlay)
            if col == 0:
                ax.set_ylabel(method_name, fontsize=10, weight='bold')
            ax.axis('off')
        
        # Method info
        ax = fig.add_subplot(gs[row, 3])
        ax.axis('off')
        
        if method_name == 'Occlusion Sensitivity':
            desc = "Most Reliable\n\nSystematically\noccludes regions\nand measures\nimpact\n\nSlow but\naccurate"
        elif method_name == 'Integrated Gradients':
            desc = "Theoretically\nSound\n\nIntegrates\ngradients along\npath from\nbaseline\n\nProvides\naxiomatic\nguarantees"
        elif method_name == 'Smooth Gradients':
            desc = "Less Noisy\n\nAverages over\nnoisy inputs\n\nReduces\nartifacts"
        elif method_name == 'Input × Gradient':
            desc = "Simple &\nEffective\n\nMultiplies\ninput by\ngradient\n\nFast"
        else:
            desc = "Basic Method\n\nRaw gradients\n\nFast but\nnoisy"
        
        ax.text(0.1, 0.5, desc, fontsize=9, verticalalignment='center',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5),
               family='monospace')
    
    # Save
    result_type = 'correct' if pred == true_label else 'incorrect'
    label_name = 'abnormal' if true_label == 1 else 'normal'
    filename = f"comprehensive_{case_id:04d}_{label_name}_{result_type}.png"
    save_path = os.path.join(output_dir, filename)
    
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    return {
        'case_id': case_id,
        'true_label': true_label,
        'prediction': pred,
        'probability': prob,
        'correct': pred == true_label,
        'filename': filename
    }

def main():
    print("="*70)
    print("🔬 COMPREHENSIVE XAI ANALYSIS - 5 METHODS")
    print("="*70)
    
    CONFIG = {
        'mrnet_path': '/Users/yuvrajpratapsingh/knee-mri-abnormality-detection/dataset/MRNet-v1.0',
        'model_path': '/Users/yuvrajpratapsingh/knee-mri-abnormality-detection/outputs/models/best_model_multiplane.pth',
        'output_dir': '/Users/yuvrajpratapsingh/knee-mri-abnormality-detection/outputs/xai_comprehensive',
        'task': 'abnormal',
    }
    
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\n🖥️  Using device: {device}")
    
    # Load dataset
    print("\n📦 Loading validation dataset...")
    val_dataset = MultiPlaneMRNetDataset(
        root_dir=CONFIG['mrnet_path'],
        task=CONFIG['task'],
        split='valid',
        use_all_slices=False
    )
    
    # Load model
    print("\n🤖 Loading multi-plane model...")
    model = MultiPlaneFusion(num_classes=1, dropout_rate=0.4)
    checkpoint = torch.load(CONFIG['model_path'], map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    print(f"✅ Model loaded from epoch {checkpoint['epoch']+1}")
    
    # Select interesting cases
    print("\n📊 Selecting 8 key cases to analyze...")
    
    # Get predictions
    all_predictions = []
    with torch.no_grad():
        model.eval()
        for idx in range(len(val_dataset)):
            batch = val_dataset[idx]
            sag = batch['sagittal'].unsqueeze(0).to(device)
            cor = batch['coronal'].unsqueeze(0).to(device)
            axi = batch['axial'].unsqueeze(0).to(device)
            label = batch['label'].item()
            
            output = model(sag, cor, axi)
            prob = torch.sigmoid(output).item()
            pred = 1 if prob > 0.5 else 0
            
            all_predictions.append({
                'idx': idx,
                'case_id': val_dataset.labels_df.iloc[idx]['case'],
                'true_label': label,
                'prediction': pred,
                'probability': prob,
                'correct': pred == label,
                'confidence': abs(prob - 0.5) * 2
            })
    
    pred_df = pd.DataFrame(all_predictions)
    
    # Select: 2 TP, 2 TN, 2 FP, 2 FN
    tp_cases = pred_df[(pred_df['prediction'] == 1) & (pred_df['true_label'] == 1)].nlargest(2, 'confidence')
    tn_cases = pred_df[(pred_df['prediction'] == 0) & (pred_df['true_label'] == 0)].nlargest(2, 'confidence')
    fp_cases = pred_df[(pred_df['prediction'] == 1) & (pred_df['true_label'] == 0)].nlargest(2, 'confidence')
    fn_cases = pred_df[(pred_df['prediction'] == 0) & (pred_df['true_label'] == 1)].nlargest(2, 'confidence')
    
    selected_indices = pd.concat([tp_cases, tn_cases, fp_cases, fn_cases])['idx'].values
    
    print(f"\n🎯 Analyzing 8 cases with 5 XAI methods each:")
    print(f"   - 2 True Positives")
    print(f"   - 2 True Negatives")
    print(f"   - 2 False Positives")
    print(f"   - 2 False Negatives")
    print(f"\nThis will take ~5-10 minutes (Occlusion is slow but worth it!)")
    
    # Analyze cases
    results = []
    
    for idx in tqdm(selected_indices, desc="Analyzing cases"):
        try:
            result = analyze_case_comprehensive(model, val_dataset, idx, device, CONFIG['output_dir'])
            results.append(result)
        except Exception as e:
            print(f"\n⚠️  Error on case {idx}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{CONFIG['output_dir']}/comprehensive_xai_summary.csv", index=False)
    
    print("\n" + "="*70)
    print("✅ COMPREHENSIVE XAI COMPLETE!")
    print("="*70)
    print(f"\n📁 Results saved to: {CONFIG['output_dir']}/")
    print(f"   - {len(results)} comprehensive visualizations")
    print(f"   - Each shows 5 different XAI methods")
    print(f"   - comprehensive_xai_summary.csv")
    print("\n💡 Compare methods to see which works best!")
    print("   Occlusion Sensitivity is usually most reliable")
    print("="*70)

if __name__ == '__main__':
    main()
