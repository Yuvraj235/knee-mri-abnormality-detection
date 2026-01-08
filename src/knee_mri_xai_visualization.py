"""
================================================================================
KNEE MRI AI - PropNet + Grad-CAM++ Visualization
================================================================================
Clean, presentation-ready explainability for knee MRI diagnosis.
Works on Mac/Windows/Linux - saves to current directory.
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.gridspec import GridSpec
import cv2
import os

# =============================================================================
# PRINT QUICK REFERENCE FOR PRESENTATION
# =============================================================================

def print_presentation_guide():
    """Print quick reference for your presentation."""
    guide = """
================================================================================
QUICK REFERENCE FOR PRESENTATION
================================================================================

WHAT IS GRAD-CAM++?
  • Shows which image regions influenced the model's decision
  • RED = high importance, BLUE = low importance
  • Better than Grad-CAM: cleaner, more focused heatmaps

WHAT IS PROPNET?
  • Propagation-based relevance scoring
  • Traces HOW MUCH each region contributes to prediction
  • Adds anatomical awareness to visualization

WHY COMBINE THEM?
  • Grad-CAM++ shows WHERE the model looks
  • PropNet shows HOW MUCH each region matters
  • Together = focused, noise-free, trustworthy explanations

ONE-PARAGRAPH EXPLANATION:
--------------------------
Our knee MRI AI uses a tri-fusion model (ResNet50 CNN + DeiT-Tiny Transformer) 
trained on 1,130 MRNet cases. To explain predictions, we combine Grad-CAM++ 
(which computes gradients to show WHERE the model focuses - red/yellow = high 
attention) with PropNet (which traces signal propagation to measure HOW MUCH 
each anatomical region contributes). Together, they create clean heatmaps that 
let radiologists instantly see what the AI examined and whether to trust it - 
supporting our doctor-in-the-loop approach where AI assists but never replaces 
clinical judgment.

================================================================================
"""
    print(guide)


# =============================================================================
# HEATMAP GENERATION (Simulated for Demo)
# =============================================================================

def generate_focused_heatmap(size=256, focus_type='acl'):
    """
    Generate realistic anatomical heatmaps for demo.
    In production, replace with actual Grad-CAM++ + PropNet output.
    """
    h, w = size, size
    heatmap = np.zeros((h, w), dtype=np.float32)
    
    if focus_type == 'acl':
        # ACL region focus (center-posterior)
        centers = [(int(h*0.5), int(w*0.5), 28), (int(h*0.55), int(w*0.45), 20)]
    elif focus_type == 'meniscus':
        # Meniscus focus (medial/lateral)
        centers = [(int(h*0.5), int(w*0.35), 22), (int(h*0.5), int(w*0.65), 22)]
    elif focus_type == 'scattered':
        # Scattered (low confidence case)
        centers = [(int(h*0.3), int(w*0.3), 15), (int(h*0.6), int(w*0.7), 18),
                   (int(h*0.7), int(w*0.4), 12)]
    else:
        # Default central focus
        centers = [(int(h*0.5), int(w*0.5), 30)]
    
    y, x = np.ogrid[:h, :w]
    for cy, cx, sigma in centers:
        gaussian = np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * sigma**2))
        heatmap += gaussian * np.random.uniform(0.7, 1.0)
    
    # Normalize
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    return heatmap


def create_synthetic_mri(size=256):
    """Create realistic-looking synthetic MRI slice for demo."""
    base = np.random.randn(size, size) * 0.12 + 0.5
    
    y, x = np.ogrid[:size, :size]
    
    # Bone structure (femur/tibia)
    bone1 = ((x - size*0.5)**2 / (size*0.3)**2 + (y - size*0.3)**2 / (size*0.25)**2) < 1
    bone2 = ((x - size*0.5)**2 / (size*0.28)**2 + (y - size*0.7)**2 / (size*0.22)**2) < 1
    base[bone1] += 0.25
    base[bone2] += 0.25
    
    # Joint space (darker)
    joint = ((x - size*0.5)**2 / (size*0.15)**2 + (y - size*0.5)**2 / (size*0.08)**2) < 1
    base[joint] -= 0.2
    
    # Soft tissue variation
    soft = ((x - size*0.35)**2 / (size*0.12)**2 + (y - size*0.5)**2 / (size*0.2)**2) < 1
    base[soft] += 0.08
    
    return np.clip(base, 0, 1)


# =============================================================================
# HEATMAP OVERLAY
# =============================================================================

def apply_heatmap(image, heatmap, alpha=0.5, threshold=0.2):
    """
    Apply clean heatmap overlay with thresholding to remove noise.
    
    Key: threshold removes scattered low-value noise for cleaner visualization.
    """
    # Normalize image to 0-255
    if len(image.shape) == 3:
        image = image.squeeze()
    img_norm = ((image - image.min()) / (image.max() - image.min() + 1e-8) * 255).astype(np.uint8)
    
    # Resize heatmap to match image
    heatmap_resized = cv2.resize(heatmap.astype(np.float32), (img_norm.shape[1], img_norm.shape[0]))
    
    # THRESHOLD: Key step to remove noise!
    heatmap_clean = np.where(heatmap_resized > threshold, heatmap_resized, 0)
    
    # Apply JET colormap
    heatmap_colored = cv2.applyColorMap((heatmap_clean * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Convert grayscale to RGB
    img_rgb = cv2.cvtColor(img_norm, cv2.COLOR_GRAY2RGB)
    
    # Smart blending - only where heatmap is active
    mask = heatmap_clean[..., np.newaxis]
    blended = img_rgb * (1 - mask * alpha) + heatmap_colored * (mask * alpha)
    
    return blended.astype(np.uint8)


# =============================================================================
# MAIN VISUALIZATION FUNCTION
# =============================================================================

def create_clinical_visualization(
    mri_slices,        # Dict: {'sagittal': array, 'coronal': array, 'axial': array}
    heatmaps,          # Dict: same structure
    prediction,        # 'NORMAL' or 'ABNORMAL'
    ground_truth,      # 'NORMAL' or 'ABNORMAL'
    confidence,        # Float 0-1
    probability,       # Abnormal probability 0-1
    case_id,
    save_path=None
):
    """
    Create clean, presentation-ready clinical visualization.
    """
    
    is_correct = prediction == ground_truth
    
    # Colors
    GREEN = '#27ae60'
    RED = '#e74c3c'
    ORANGE = '#f39c12'
    GRAY = '#7f8c8d'
    DARK = '#2c3e50'
    
    status_color = GREEN if is_correct else RED
    status_text = '✓ CORRECT' if is_correct else '✗ INCORRECT'
    
    # Create figure
    fig = plt.figure(figsize=(16, 9), facecolor='white')
    
    # Title
    fig.suptitle(f'Case {case_id}  •  {status_text}', 
                fontsize=22, fontweight='bold', color=status_color, y=0.96)
    fig.text(0.5, 0.91, 'PropNet + Grad-CAM++ Visualization',
            fontsize=12, ha='center', color=GRAY, style='italic')
    
    # Layout
    gs = GridSpec(2, 4, height_ratios=[3, 1.2], width_ratios=[1, 1, 1, 0.6],
                  hspace=0.15, wspace=0.12, top=0.88, bottom=0.08, left=0.05, right=0.95)
    
    # === TOP ROW: MRI Views ===
    views = ['sagittal', 'coronal', 'axial']
    titles = ['Sagittal View', 'Coronal View', 'Axial View']
    
    for idx, (view, title) in enumerate(zip(views, titles)):
        ax = fig.add_subplot(gs[0, idx])
        overlay = apply_heatmap(mri_slices[view], heatmaps[view], alpha=0.5)
        ax.imshow(overlay)
        ax.set_title(title, fontsize=14, fontweight='bold', pad=10, color=DARK)
        ax.axis('off')
    
    # === TOP RIGHT: Legend ===
    ax_legend = fig.add_subplot(gs[0, 3])
    ax_legend.axis('off')
    ax_legend.set_xlim(0, 1)
    ax_legend.set_ylim(0, 1)
    
    ax_legend.text(0.5, 0.95, 'Attention Guide', fontsize=13, fontweight='bold',
                  ha='center', va='top', color=DARK)
    
    # Colorbar
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    ax_cbar = ax_legend.inset_axes([0.1, 0.75, 0.8, 0.08])
    ax_cbar.imshow(gradient, aspect='auto', cmap='jet')
    ax_cbar.set_xticks([0, 128, 255])
    ax_cbar.set_xticklabels(['Low', 'Med', 'High'], fontsize=9)
    ax_cbar.set_yticks([])
    
    # Legend items
    legend_items = [('Red/Yellow:', 'High attention'),
                   ('Green/Cyan:', 'Medium attention'),
                   ('Blue/None:', 'Low/No attention')]
    
    for i, (label, desc) in enumerate(legend_items):
        y = 0.58 - i * 0.1
        ax_legend.text(0.1, y, label, fontsize=10, fontweight='bold', color=DARK)
        ax_legend.text(0.5, y, desc, fontsize=10, color=GRAY)
    
    ax_legend.text(0.5, 0.18, 'Method:', fontsize=10, fontweight='bold', ha='center', color=DARK)
    ax_legend.text(0.5, 0.08, 'PropNet + Grad-CAM++', fontsize=9, ha='center', color=GRAY)
    
    # === BOTTOM ROW: Metrics ===
    ax_bottom = fig.add_subplot(gs[1, :])
    ax_bottom.axis('off')
    ax_bottom.set_xlim(0, 1)
    ax_bottom.set_ylim(0, 1)
    
    # Background box
    bg = FancyBboxPatch((0.02, 0.1), 0.96, 0.8, boxstyle="round,pad=0.02",
                        facecolor='#f8f9fa', edgecolor='#dee2e6', linewidth=2,
                        transform=ax_bottom.transAxes)
    ax_bottom.add_patch(bg)
    
    # Metrics
    pred_color = RED if prediction == 'ABNORMAL' else GREEN
    gt_color = RED if ground_truth == 'ABNORMAL' else GREEN
    
    metrics = [
        ('Prediction', prediction, pred_color),
        ('Ground Truth', ground_truth, gt_color),
        ('Confidence', f'{confidence*100:.1f}%', DARK),
        ('Abnormal Prob', f'{probability*100:.1f}%', DARK),
    ]
    
    for i, (label, value, color) in enumerate(metrics):
        x = 0.1 + i * 0.15
        ax_bottom.text(x, 0.7, label, fontsize=11, color=GRAY, ha='center',
                      transform=ax_bottom.transAxes)
        ax_bottom.text(x, 0.45, value, fontsize=16, fontweight='bold', color=color,
                      ha='center', transform=ax_bottom.transAxes)
    
    # Confidence bar
    bar_x, bar_w = 0.68, 0.25
    ax_bottom.text(bar_x, 0.7, 'Confidence Level:', fontsize=11, color=GRAY,
                  transform=ax_bottom.transAxes)
    
    bar_bg = Rectangle((bar_x, 0.4), bar_w, 0.2, transform=ax_bottom.transAxes,
                       facecolor='#e9ecef', edgecolor='#dee2e6')
    ax_bottom.add_patch(bar_bg)
    
    conf_color = GREEN if confidence > 0.6 else ORANGE if confidence > 0.4 else RED
    bar_fill = Rectangle((bar_x, 0.4), bar_w * confidence, 0.2,
                         transform=ax_bottom.transAxes, facecolor=conf_color)
    ax_bottom.add_patch(bar_fill)
    
    # Recommendation
    if confidence > 0.7:
        rec = "✓ High confidence — Suitable for screening assistance"
        rec_color = GREEN
    elif confidence > 0.5:
        rec = "⚠ Medium confidence — Radiologist review recommended"
        rec_color = ORANGE
    else:
        rec = "⚠ Low confidence — Manual review required"
        rec_color = RED
    
    ax_bottom.text(0.5, 0.12, rec, fontsize=13, fontweight='bold', color=rec_color,
                  ha='center', transform=ax_bottom.transAxes)
    
    # Save
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"  ✓ Saved: {save_path}")
    
    plt.close()
    return fig


# =============================================================================
# BEFORE vs AFTER COMPARISON
# =============================================================================

def create_comparison_figure(save_path=None):
    """Show improvement from noisy to clean visualization."""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor='white')
    fig.suptitle('Visualization Improvement: Before vs After', fontsize=18, fontweight='bold')
    
    # BEFORE: Noisy
    ax1 = axes[0]
    ax1.set_title('BEFORE: Noisy & Cluttered', fontsize=14, color='#e74c3c', pad=10)
    
    np.random.seed(42)
    noisy = np.random.rand(200, 200) * 0.6
    y, x = np.ogrid[:200, :200]
    for _ in range(6):
        cx, cy = np.random.randint(20, 180, 2)
        noisy += np.exp(-((x-cx)**2 + (y-cy)**2) / (2*20**2)) * 0.4
    
    ax1.imshow(noisy, cmap='jet')
    ax1.axis('off')
    ax1.text(0.5, -0.05, 'Scattered attention, hard to interpret', 
            fontsize=10, ha='center', transform=ax1.transAxes, color='#666')
    
    # AFTER: Clean
    ax2 = axes[1]
    ax2.set_title('AFTER: Clean & Focused', fontsize=14, color='#27ae60', pad=10)
    
    clean = np.zeros((200, 200))
    centers = [(100, 100, 30), (90, 110, 20)]
    for cy, cx, sigma in centers:
        clean += np.exp(-((x-cx)**2 + (y-cy)**2) / (2*sigma**2))
    clean = clean / clean.max()
    clean = np.where(clean > 0.2, clean, 0)  # Threshold!
    
    ax2.imshow(clean, cmap='jet')
    ax2.axis('off')
    ax2.text(0.5, -0.05, 'Focused attention, easy to interpret',
            fontsize=10, ha='center', transform=ax2.transAxes, color='#666')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"  ✓ Saved: {save_path}")
    
    plt.close()


# =============================================================================
# METHOD EXPLANATION FIGURE
# =============================================================================

def create_method_figure(save_path=None):
    """Explain how PropNet + Grad-CAM++ work together."""
    
    fig = plt.figure(figsize=(16, 8), facecolor='white')
    fig.suptitle('How PropNet + Grad-CAM++ Work Together', fontsize=20, fontweight='bold')
    
    gs = GridSpec(2, 3, hspace=0.35, wspace=0.25, top=0.88, bottom=0.08)
    
    # Step 1: Input
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title('1. Input MRI', fontsize=14, fontweight='bold')
    mri = create_synthetic_mri(150)
    ax1.imshow(mri, cmap='gray')
    ax1.axis('off')
    ax1.text(0.5, -0.12, 'Sagittal/Coronal/Axial\nslices fed to model',
            fontsize=10, ha='center', transform=ax1.transAxes, color='#666')
    
    # Step 2: Model
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_title('2. Tri-Fusion Model', fontsize=14, fontweight='bold')
    ax2.axis('off')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    
    boxes = [
        (0.05, 0.7, 0.4, 0.2, 'ResNet50\n(CNN)', '#3498db'),
        (0.55, 0.7, 0.4, 0.2, 'DeiT-Tiny\n(Transformer)', '#9b59b6'),
        (0.2, 0.35, 0.6, 0.2, 'Fusion Layer', '#2ecc71'),
        (0.3, 0.05, 0.4, 0.15, 'Prediction', '#e74c3c'),
    ]
    
    for x, y, w, h, label, color in boxes:
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                              facecolor=color, alpha=0.8, edgecolor='white', linewidth=2,
                              transform=ax2.transAxes)
        ax2.add_patch(rect)
        ax2.text(x + w/2, y + h/2, label, fontsize=9, ha='center', va='center',
                transform=ax2.transAxes, fontweight='bold', color='white')
    
    # Arrows
    ax2.annotate('', xy=(0.5, 0.55), xytext=(0.25, 0.7),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5), transform=ax2.transAxes)
    ax2.annotate('', xy=(0.5, 0.55), xytext=(0.75, 0.7),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5), transform=ax2.transAxes)
    ax2.annotate('', xy=(0.5, 0.2), xytext=(0.5, 0.35),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5), transform=ax2.transAxes)
    
    # Step 3: XAI Methods
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_title('3. Explainability', fontsize=14, fontweight='bold')
    ax3.axis('off')
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    
    # Grad-CAM++ box
    rect1 = FancyBboxPatch((0.05, 0.55), 0.9, 0.35, boxstyle="round,pad=0.02",
                           facecolor='#e67e22', alpha=0.2, edgecolor='#e67e22', linewidth=2,
                           transform=ax3.transAxes)
    ax3.add_patch(rect1)
    ax3.text(0.1, 0.8, 'Grad-CAM++', fontsize=11, fontweight='bold', color='#e67e22',
            transform=ax3.transAxes)
    ax3.text(0.1, 0.62, '• Gradient-based attention\n• Shows WHERE model looks',
            fontsize=9, color='#333', transform=ax3.transAxes)
    
    # PropNet box
    rect2 = FancyBboxPatch((0.05, 0.1), 0.9, 0.35, boxstyle="round,pad=0.02",
                           facecolor='#16a085', alpha=0.2, edgecolor='#16a085', linewidth=2,
                           transform=ax3.transAxes)
    ax3.add_patch(rect2)
    ax3.text(0.1, 0.35, 'PropNet', fontsize=11, fontweight='bold', color='#16a085',
            transform=ax3.transAxes)
    ax3.text(0.1, 0.17, '• Propagation-based\n• Shows HOW MUCH contribution',
            fontsize=9, color='#333', transform=ax3.transAxes)
    
    # Bottom: Combined result
    ax_bottom = fig.add_subplot(gs[1, :])
    ax_bottom.set_title('4. Combined Result: Clean, Focused Attention Maps', 
                       fontsize=14, fontweight='bold', pad=10)
    
    # Three view heatmaps
    combined = np.zeros((150, 500))
    y, x = np.ogrid[:150, :500]
    
    for cx in [85, 250, 415]:
        combined += np.exp(-((x - cx)**2 + (y - 75)**2) / (2 * 25**2))
    
    combined = combined / combined.max()
    combined = np.where(combined > 0.15, combined, 0)
    
    ax_bottom.imshow(combined, cmap='jet', aspect='auto')
    ax_bottom.axis('off')
    
    # Labels
    for i, label in enumerate(['Sagittal', 'Coronal', 'Axial']):
        ax_bottom.text((85 + i*165)/500, -0.08, label, fontsize=11, fontweight='bold',
                      ha='center', transform=ax_bottom.transAxes)
    
    # Key insight
    fig.text(0.5, 0.02, 
            'Key: Combining PropNet + Grad-CAM++ gives focused maps showing WHERE the model looks AND HOW MUCH each region matters',
            fontsize=11, ha='center', style='italic', color='#2c3e50',
            bbox=dict(boxstyle='round', facecolor='#ecf0f1', edgecolor='#bdc3c7'))
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"  ✓ Saved: {save_path}")
    
    plt.close()


# =============================================================================
# RUN DEMO
# =============================================================================

def run_demo():
    """Generate all demo figures."""
    
    print_presentation_guide()
    
    # Get current directory for saving
    output_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"\n{'='*60}")
    print("GENERATING DEMO VISUALIZATIONS")
    print(f"{'='*60}")
    print(f"Output folder: {output_dir}\n")
    
    # Create synthetic data
    mri_slices = {
        'sagittal': create_synthetic_mri(256),
        'coronal': create_synthetic_mri(256),
        'axial': create_synthetic_mri(256),
    }
    
    # === Example 1: Correct Normal ===
    print("[1/5] Correct Normal Prediction (High Confidence)")
    heatmaps_1 = {v: generate_focused_heatmap(256, 'acl') for v in ['sagittal', 'coronal', 'axial']}
    create_clinical_visualization(
        mri_slices=mri_slices, heatmaps=heatmaps_1,
        prediction='NORMAL', ground_truth='NORMAL',
        confidence=0.82, probability=0.18, case_id='1136',
        save_path=os.path.join(output_dir, 'example_1_correct_normal.png')
    )
    
    # === Example 2: Correct Abnormal ===
    print("[2/5] Correct Abnormal Prediction (High Confidence)")
    heatmaps_2 = {v: generate_focused_heatmap(256, 'meniscus') for v in ['sagittal', 'coronal', 'axial']}
    create_clinical_visualization(
        mri_slices=mri_slices, heatmaps=heatmaps_2,
        prediction='ABNORMAL', ground_truth='ABNORMAL',
        confidence=0.89, probability=0.91, case_id='1098',
        save_path=os.path.join(output_dir, 'example_2_correct_abnormal.png')
    )
    
    # === Example 3: Incorrect (False Positive) ===
    print("[3/5] Incorrect Prediction (Low Confidence)")
    heatmaps_3 = {v: generate_focused_heatmap(256, 'scattered') for v in ['sagittal', 'coronal', 'axial']}
    create_clinical_visualization(
        mri_slices=mri_slices, heatmaps=heatmaps_3,
        prediction='ABNORMAL', ground_truth='NORMAL',
        confidence=0.44, probability=0.72, case_id='1137',
        save_path=os.path.join(output_dir, 'example_3_incorrect.png')
    )
    
    # === Example 4: Before vs After ===
    print("[4/5] Before vs After Comparison")
    create_comparison_figure(
        save_path=os.path.join(output_dir, 'example_4_before_after.png')
    )
    
    # === Example 5: Method Explanation ===
    print("[5/5] Method Explanation Figure")
    create_method_figure(
        save_path=os.path.join(output_dir, 'example_5_method_explanation.png')
    )
    
    print(f"\n{'='*60}")
    print("DONE! All figures saved to:")
    print(f"{'='*60}")
    print(f"  {output_dir}/")
    print("    • example_1_correct_normal.png")
    print("    • example_2_correct_abnormal.png")
    print("    • example_3_incorrect.png")
    print("    • example_4_before_after.png")
    print("    • example_5_method_explanation.png")
    print(f"{'='*60}\n")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    run_demo()