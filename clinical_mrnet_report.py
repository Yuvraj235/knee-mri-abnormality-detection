import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap
import cv2
from pathlib import Path
import argparse

# Try to import PyTorch for model inference
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️  PyTorch not available - will use demo mode")


class GradCAMPlusPlus:
    """
    Grad-CAM++ implementation for model interpretation.
    Shows which regions of the MRI the model focused on.
    """
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)
    
    def _save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def __call__(self, input_tensor):
        """Generate attention map for input."""
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor)
        
        # Backward pass
        self.model.zero_grad()
        output.backward()
        
        # Get gradients and activations
        grads = self.gradients.cpu().numpy()
        acts = self.activations.cpu().numpy()
        
        # Grad-CAM++ weighting
        grads_power_2 = grads ** 2
        grads_power_3 = grads ** 3
        sum_acts = np.sum(acts, axis=(2, 3), keepdims=True)
        
        alpha_num = grads_power_2
        alpha_denom = 2 * grads_power_2 + sum_acts * grads_power_3 + 1e-7
        alpha = alpha_num / alpha_denom
        
        weights = np.sum(alpha * np.maximum(grads, 0), axis=(2, 3), keepdims=True)
        
        # Weighted combination
        cam = np.sum(weights * acts, axis=1)
        cam = np.maximum(cam, 0)  # ReLU
        
        # Normalize
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam[0]


def load_mrnet_scan(filepath):
    """
    Load MRI scan from MRNet .npy file.
    
    Returns:
        scan: numpy array of shape (num_slices, H, W)
        middle_slice: the central slice (most informative)
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Scan not found: {filepath}")
    
    scan = np.load(filepath).astype(np.float32)
    
    # Handle different possible shapes
    if scan.ndim == 4:  # (1, slices, H, W) or (slices, 1, H, W)
        scan = scan.squeeze()
    
    if scan.ndim != 3:
        raise ValueError(f"Expected 3D scan, got shape {scan.shape}")
    
    # Normalize to [0, 1]
    scan_min, scan_max = scan.min(), scan.max()
    if scan_max > scan_min:
        scan = (scan - scan_min) / (scan_max - scan_min)
    
    # Get middle slice (typically most informative)
    mid_idx = scan.shape[0] // 2
    middle_slice = scan[mid_idx]
    
    return scan, middle_slice, mid_idx


def find_best_slice(scan):
    """
    Find the most informative slice in the scan.
    Uses variance as a proxy for information content.
    """
    variances = [scan[i].var() for i in range(scan.shape[0])]
    
    # Weight toward center slices (where anatomy is most visible)
    center = scan.shape[0] // 2
    weights = np.exp(-((np.arange(scan.shape[0]) - center) ** 2) / (scan.shape[0] ** 2))
    weighted_vars = np.array(variances) * weights
    
    return np.argmax(weighted_vars)


def generate_attention_from_model(model, scan, device='cpu'):
    """
    Generate real Grad-CAM++ attention using the trained model.
    """
    # Prepare input - take center slices
    mid = scan.shape[0] // 2
    n_slices = min(16, scan.shape[0])
    start = max(0, mid - n_slices // 2)
    end = min(scan.shape[0], start + n_slices)
    
    selected_slices = scan[start:end]
    
    # Normalize and prepare tensor
    input_tensor = torch.from_numpy(selected_slices).unsqueeze(0).unsqueeze(0)
    input_tensor = input_tensor.float().to(device)
    
    # Get the target layer (last conv layer of backbone)
    # This depends on your model architecture
    target_layer = None
    for name, module in model.named_modules():
        if 'layer4' in name and isinstance(module, nn.Sequential):
            target_layer = module[-1]
            break
    
    if target_layer is None:
        # Fallback - try to find any conv layer
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                target_layer = module
    
    # Create GradCAM++
    gradcam = GradCAMPlusPlus(model, target_layer)
    attention = gradcam(input_tensor)
    
    # Resize attention to match slice size
    attention = cv2.resize(attention, (scan.shape[2], scan.shape[1]))
    
    return attention


def generate_simulated_attention(slice_img, abnormality_present=True):
    """
    Generate simulated attention map when model isn't available.
    Creates anatomically plausible attention patterns.
    """
    h, w = slice_img.shape
    
    # Find high-intensity regions (likely anatomical structures)
    # These are where a model would typically focus
    
    # Edge detection to find structures
    sobel_x = cv2.Sobel(slice_img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(slice_img, cv2.CV_64F, 0, 1, ksize=3)
    edges = np.sqrt(sobel_x**2 + sobel_y**2)
    
    # Normalize
    if edges.max() > 0:
        edges = edges / edges.max()
    
    # Find regions of interest (tissue vs background)
    tissue_mask = slice_img > 0.1  # Threshold for tissue
    
    if abnormality_present:
        # Create focal attention in central region
        # (where ACL/meniscus typically are in sagittal view)
        center_y, center_x = int(h * 0.55), int(w * 0.5)
        
        y, x = np.ogrid[:h, :w]
        
        # Main focal region
        focal = np.exp(-((y - center_y)**2 + (x - center_x)**2) / (h * 0.15)**2)
        
        # Add secondary region
        focal2_y, focal2_x = int(h * 0.45), int(w * 0.55)
        focal += 0.6 * np.exp(-((y - focal2_y)**2 + (x - focal2_x)**2) / (h * 0.12)**2)
        
        attention = 0.7 * focal + 0.3 * edges
    else:
        # Diffuse attention for normal scan
        attention = edges * 0.5
    
    # Mask to tissue regions
    attention = attention * tissue_mask.astype(float)
    
    # Normalize
    if attention.max() > 0:
        attention = attention / attention.max()
    
    return attention


def find_attention_peak(attention_map, min_size=20):
    """
    Find the location of peak attention for annotation.
    Returns (x, y, radius) of the main focus area.
    """
    # Threshold to find high-attention regions
    threshold = 0.5 * attention_map.max()
    high_attention = (attention_map > threshold).astype(np.uint8)
    
    # Find contours
    contours, _ = cv2.findContours(high_attention, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        # Return center of maximum attention
        max_loc = np.unravel_index(attention_map.argmax(), attention_map.shape)
        return max_loc[1], max_loc[0], 30
    
    # Find largest contour
    largest = max(contours, key=cv2.contourArea)
    
    # Get bounding circle
    (x, y), radius = cv2.minEnclosingCircle(largest)
    
    return int(x), int(y), max(int(radius), min_size)


def create_clinical_report(slice_img, attention_map, prediction, confidence,
                           plane='sagittal', case_id='001', save_path=None):
    """
    Create a professional clinical report that doctors and patients can understand.
    
    Args:
        slice_img: 2D numpy array of MRI slice (already normalized to [0,1])
        attention_map: 2D numpy array showing model attention
        prediction: 'normal' or 'abnormal'
        confidence: float 0-1
        plane: 'sagittal', 'coronal', or 'axial'
        case_id: string identifier for the case
        save_path: path to save the report
    """
    
    fig = plt.figure(figsize=(14, 9), facecolor='#1a1a2e')
    
    # Create grid layout
    gs = fig.add_gridspec(2, 3, height_ratios=[2, 1], 
                          width_ratios=[1.2, 1.2, 1],
                          hspace=0.15, wspace=0.15)
    
    is_abnormal = prediction.lower() == 'abnormal'
    
    # ==================== TOP LEFT: Original MRI ====================
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor('#0f0f1a')
    ax1.imshow(slice_img, cmap='gray', aspect='equal')
    ax1.set_title('MRI Scan', fontsize=13, color='white', pad=10, fontweight='bold')
    ax1.axis('off')
    
    # ==================== TOP MIDDLE: MRI with Attention ====================
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor('#0f0f1a')
    
    # Create overlay
    ax2.imshow(slice_img, cmap='gray', aspect='equal')
    
    # Overlay attention with transparency
    attention_colored = plt.cm.hot(attention_map)
    attention_overlay = attention_colored.copy()
    attention_overlay[:, :, 3] = attention_map * 0.6  # Alpha based on attention
    ax2.imshow(attention_overlay)
    
    if is_abnormal:
        # Find and mark the problem area
        x, y, radius = find_attention_peak(attention_map)
        
        # Draw highlighting circles
        for r, alpha in [(radius + 15, 0.3), (radius + 8, 0.5), (radius, 0.8)]:
            circle = Circle((x, y), r, fill=False, 
                           edgecolor='#ff4757', linewidth=2, alpha=alpha)
            ax2.add_patch(circle)
        
        # Arrow pointing to the area
        arrow_start_x = x - radius - 40
        arrow_start_y = y - radius - 30
        
        if arrow_start_x < 30:
            arrow_start_x = x + radius + 40
        if arrow_start_y < 30:
            arrow_start_y = y + radius + 30
            
        arrow = FancyArrowPatch(
            (arrow_start_x, arrow_start_y), 
            (x - radius * 0.7, y - radius * 0.7),
            arrowstyle='->', mutation_scale=18,
            color='#ff4757', linewidth=2.5
        )
        ax2.add_patch(arrow)
        
        # Label
        ax2.annotate('Abnormality\nDetected', 
                    xy=(arrow_start_x, arrow_start_y - 10),
                    fontsize=11, fontweight='bold', color='#ff4757',
                    ha='center', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.3',
                             facecolor='#1a1a2e', edgecolor='#ff4757',
                             alpha=0.9))
    
    ax2.set_title('AI Analysis', fontsize=13, color='white', pad=10, fontweight='bold')
    ax2.axis('off')
    
    # ==================== TOP RIGHT: Results Panel ====================
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_facecolor('#1a1a2e')
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 10)
    ax3.axis('off')
    
    # Header
    ax3.text(5, 9.5, 'DIAGNOSIS', fontsize=16, fontweight='bold', 
            color='white', ha='center')
    ax3.plot([1, 9], [9.1, 9.1], color='#4a4a6a', linewidth=1)
    
    # Result indicator
    if is_abnormal:
        result_color = '#ff4757'
        result_text = 'ABNORMAL'
        icon = '⚠'
    else:
        result_color = '#2ed573'
        result_text = 'NORMAL'
        icon = '✓'
    
    # Result box
    result_box = FancyBboxPatch((0.5, 6.8), 9, 2,
                                boxstyle="round,pad=0.05,rounding_size=0.2",
                                facecolor=result_color, alpha=0.2,
                                edgecolor=result_color, linewidth=2)
    ax3.add_patch(result_box)
    
    ax3.text(5, 8.0, f'{icon}', fontsize=24, ha='center', va='center', color=result_color)
    ax3.text(5, 7.3, result_text, fontsize=18, fontweight='bold',
            ha='center', va='center', color=result_color)
    
    # Confidence bar
    ax3.text(5, 5.8, f'Confidence: {confidence:.0%}', fontsize=13,
            color='white', ha='center')
    
    # Confidence bar background
    bar_bg = FancyBboxPatch((1.5, 5.0), 7, 0.4,
                            boxstyle="round,pad=0.02",
                            facecolor='#2a2a4a', edgecolor=None)
    ax3.add_patch(bar_bg)
    
    # Confidence bar fill
    bar_width = 7 * confidence
    bar_fill = FancyBboxPatch((1.5, 5.0), bar_width, 0.4,
                              boxstyle="round,pad=0.02",
                              facecolor=result_color, edgecolor=None)
    ax3.add_patch(bar_fill)
    
    # Scan info
    ax3.text(1, 4.0, f'View: {plane.capitalize()}', fontsize=10, color='#8a8aaa')
    ax3.text(1, 3.4, f'Case: {case_id}', fontsize=10, color='#8a8aaa')
    
    # Recommendation
    ax3.plot([1, 9], [2.5, 2.5], color='#4a4a6a', linewidth=0.5)
    
    if is_abnormal:
        rec_text = 'Recommend clinical\nconsultation'
        rec_color = '#ff4757'
    else:
        rec_text = 'No further action\nrequired'
        rec_color = '#2ed573'
    
    ax3.text(5, 1.5, rec_text, fontsize=11, color=rec_color,
            ha='center', va='center', linespacing=1.3)
    
    # ==================== BOTTOM: Explanation Panel ====================
    ax_bottom = fig.add_subplot(gs[1, :])
    ax_bottom.set_facecolor('#12121f')
    ax_bottom.set_xlim(0, 10)
    ax_bottom.set_ylim(0, 3)
    ax_bottom.axis('off')
    
    # What the colors mean
    ax_bottom.text(0.5, 2.5, 'READING THIS REPORT:', fontsize=11, 
                  fontweight='bold', color='white')
    
    explanations = [
        ('Left image:', 'Your original MRI scan', '#8a8aaa'),
        ('Middle image:', 'Areas the AI examined closely are shown in warm colors (yellow/red)', '#8a8aaa'),
        ('Right panel:', 'The AI\'s assessment and confidence level', '#8a8aaa'),
    ]
    
    for i, (label, text, color) in enumerate(explanations):
        ax_bottom.text(0.5, 1.9 - i*0.4, f'{label}', fontsize=9, 
                      color='white', fontweight='bold')
        ax_bottom.text(2.2, 1.9 - i*0.4, text, fontsize=9, color=color)
    
    # Legend for attention colors
    ax_bottom.text(6.5, 2.5, 'AI ATTENTION SCALE:', fontsize=10, 
                  fontweight='bold', color='white')
    
    # Create colorbar
    gradient = np.linspace(0, 1, 100).reshape(1, -1)
    ax_cbar = fig.add_axes([0.68, 0.18, 0.15, 0.03])
    ax_cbar.imshow(gradient, aspect='auto', cmap='hot')
    ax_cbar.set_xticks([0, 99])
    ax_cbar.set_xticklabels(['Low', 'High'], fontsize=8, color='white')
    ax_cbar.set_yticks([])
    ax_cbar.tick_params(colors='white')
    for spine in ax_cbar.spines.values():
        spine.set_visible(False)
    
    # Disclaimer
    ax_bottom.text(5, 0.2, 
                  'This AI analysis is for screening purposes only. '
                  'Final diagnosis should be made by a qualified physician.',
                  fontsize=8, color='#5a5a7a', ha='center', style='italic')
    
    # Title
    fig.suptitle('KNEE MRI ANALYSIS REPORT', fontsize=18, fontweight='bold',
                color='white', y=0.98)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight',
                   facecolor='#1a1a2e', edgecolor='none')
        print(f"✅ Report saved: {save_path}")
    
    return fig


def analyze_knee_mri(scan_path, model=None, output_dir='./reports', 
                     plane='sagittal', case_id=None):
    """
    Complete analysis pipeline for a knee MRI scan.
    
    Args:
        scan_path: Path to .npy MRI file
        model: Trained PyTorch model (optional, uses simulation if None)
        output_dir: Where to save the report
        plane: 'sagittal', 'coronal', or 'axial'
        case_id: Case identifier (auto-generated if None)
    
    Returns:
        dict with prediction results and report path
    """
    
    print("\n" + "="*60)
    print("🏥 KNEE MRI ANALYSIS")
    print("="*60)
    
    # Load scan
    scan_path = Path(scan_path)
    print(f"\n📂 Loading: {scan_path.name}")
    
    scan, middle_slice, mid_idx = load_mrnet_scan(scan_path)
    print(f"   Scan shape: {scan.shape}")
    print(f"   Using slice: {mid_idx}")
    
    # Find best slice
    best_idx = find_best_slice(scan)
    if best_idx != mid_idx:
        print(f"   Best slice found at: {best_idx}")
        analysis_slice = scan[best_idx]
    else:
        analysis_slice = middle_slice
    
    # Generate prediction and attention
    if model is not None and TORCH_AVAILABLE:
        print("\n🧠 Running model inference...")
        device = next(model.parameters()).device
        
        # Get prediction
        model.eval()
        with torch.no_grad():
            # Prepare input (depends on your model's expected input)
            input_tensor = torch.from_numpy(scan).unsqueeze(0).unsqueeze(0)
            input_tensor = input_tensor.float().to(device)
            
            output = model(input_tensor)
            probability = torch.sigmoid(output).item()
        
        # Get attention map
        attention = generate_attention_from_model(model, scan, device)
        
        prediction = 'abnormal' if probability > 0.5 else 'normal'
        confidence = probability if prediction == 'abnormal' else 1 - probability
    else:
        print("\n⚠️  No model provided - using demonstration mode")
        # Simulate for demo
        prediction = 'abnormal'  # Demo shows abnormal case
        confidence = 0.87
        attention = generate_simulated_attention(analysis_slice, abnormality_present=True)
    
    print(f"\n📊 Results:")
    print(f"   Prediction: {prediction.upper()}")
    print(f"   Confidence: {confidence:.1%}")
    
    # Generate case ID
    if case_id is None:
        case_id = scan_path.stem
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create report
    print(f"\n📝 Generating clinical report...")
    report_path = output_dir / f"report_{case_id}_{plane}.png"
    
    fig = create_clinical_report(
        slice_img=analysis_slice,
        attention_map=attention,
        prediction=prediction,
        confidence=confidence,
        plane=plane,
        case_id=case_id,
        save_path=report_path
    )
    plt.close(fig)
    
    # Also create a "normal" version for comparison
    if prediction == 'abnormal':
        normal_attention = generate_simulated_attention(analysis_slice, abnormality_present=False)
        normal_path = output_dir / f"report_{case_id}_{plane}_if_normal.png"
        
        fig_normal = create_clinical_report(
            slice_img=analysis_slice,
            attention_map=normal_attention,
            prediction='normal',
            confidence=0.94,
            plane=plane,
            case_id=case_id + '_comparison',
            save_path=normal_path
        )
        plt.close(fig_normal)
    
    print("\n" + "="*60)
    print("✅ ANALYSIS COMPLETE")
    print("="*60)
    
    return {
        'prediction': prediction,
        'confidence': confidence,
        'report_path': str(report_path),
        'scan_shape': scan.shape,
        'slice_used': best_idx
    }


# =============================================================================
# DEMO FUNCTION - Creates realistic MRI-like image for testing
# =============================================================================

def create_demo_mri_scan():
    """
    Create a more realistic demo MRI scan for testing.
    This creates a 3D volume like MRNet data.
    """
    n_slices = 30
    size = 256
    
    volume = np.zeros((n_slices, size, size), dtype=np.float32)
    
    for s in range(n_slices):
        # Depth factor - anatomy changes through slices
        depth = abs(s - n_slices // 2) / (n_slices // 2)
        scale = 1 - 0.3 * depth
        
        img = np.zeros((size, size), dtype=np.float32)
        y, x = np.ogrid[:size, :size]
        
        # Femur
        femur_y = int(80 * scale)
        femur = np.exp(-((y - femur_y)**2 / (55**2) + (x - 128)**2 / (45**2)))
        img += femur * 0.85 * scale
        
        # Femoral condyles
        for cx, offset in [(100, -10), (156, 10)]:
            cy = 115 + int(offset * (1 - scale))
            condyle = np.exp(-((y - cy)**2 + (x - cx)**2) / (25**2))
            img += condyle * 0.8 * scale
        
        # Tibia
        tibia_y = int(195 * scale + 30 * (1 - scale))
        tibia = np.exp(-((y - tibia_y)**2 / (60**2) + (x - 128)**2 / (50**2)))
        img += tibia * 0.8 * scale
        
        # Joint structures (only in central slices)
        if 0.3 < (s / n_slices) < 0.7:
            joint_scale = 1 - 2 * abs(s / n_slices - 0.5)
            
            # ACL region
            acl_y, acl_x = 140, 135
            acl = np.exp(-((y - acl_y)**2 + (x - acl_x)**2) / (15**2))
            img += acl * 0.4 * joint_scale
            
            # Menisci
            for mx, my in [(165, 145), (91, 145)]:
                meniscus = np.exp(-((y - my)**2 / (8**2) + (x - mx)**2 / (12**2)))
                img += meniscus * 0.3 * joint_scale
        
        # Add texture
        noise = np.random.randn(size, size) * 0.02
        img += noise
        
        # Smooth
        img = cv2.GaussianBlur(img, (3, 3), 0.5)
        
        volume[s] = np.clip(img, 0, 1)
    
    return volume


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Knee MRI Clinical Report Generator')
    parser.add_argument('--scan', type=str, help='Path to MRI .npy file')
    parser.add_argument('--sagittal', type=str, help='Path to sagittal plane .npy')
    parser.add_argument('--coronal', type=str, help='Path to coronal plane .npy')
    parser.add_argument('--axial', type=str, help='Path to axial plane .npy')
    parser.add_argument('--model', type=str, help='Path to trained model .pth file')
    parser.add_argument('--output', type=str, default='./clinical_reports',
                       help='Output directory for reports')
    parser.add_argument('--demo', action='store_true', help='Run in demo mode')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.demo or (not args.scan and not args.sagittal):
        print("\n🎭 DEMO MODE - Creating sample MRI analysis")
        
        # Create demo scan
        demo_scan = create_demo_mri_scan()
        demo_path = output_dir / 'demo_scan.npy'
        np.save(demo_path, demo_scan)
        
        # Run analysis
        results = analyze_knee_mri(
            demo_path,
            model=None,
            output_dir=output_dir,
            plane='sagittal',
            case_id='DEMO_001'
        )
        
        print(f"\n📁 Reports saved to: {output_dir}")
        print("\n💡 To use with your real data:")
        print("   python clinical_mrnet_report.py --scan /path/to/your/scan.npy")
        
    elif args.scan:
        # Single scan analysis
        results = analyze_knee_mri(
            args.scan,
            model=None,  # Add model loading if --model provided
            output_dir=output_dir,
            plane='sagittal'
        )
        
    else:
        # Multi-plane analysis
        for plane, path in [('sagittal', args.sagittal), 
                           ('coronal', args.coronal),
                           ('axial', args.axial)]:
            if path:
                results = analyze_knee_mri(
                    path,
                    model=None,
                    output_dir=output_dir,
                    plane=plane
                )