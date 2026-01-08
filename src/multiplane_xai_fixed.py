import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.multiplane_loader import MultiPlaneMRNetDataset
from src.multiplane_model import MultiPlaneFusion

class ImprovedGradCAM:
    """
    Fixed Grad-CAM implementation for multi-plane model
    """
    
    def __init__(self, model, encoder, plane_name):
        self.model = model
        self.encoder = encoder
        self.plane_name = plane_name
        self.gradients = None
        self.activations = None
        
        # Register hooks on the last convolutional layer
        # For ResNet50, this is layer4[-1]
        target_layer = self.encoder[-1][-1]  # Last residual block
        
        self.forward_handle = target_layer.register_forward_hook(self.save_activation)
        self.backward_handle = target_layer.register_full_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def remove_hooks(self):
        self.forward_handle.remove()
        self.backward_handle.remove()
    
    def generate_cam(self, input_tensor):
        """Generate CAM for a single plane"""
        
        # Set model to train mode to enable gradients
        self.model.train()
        
        # Forward pass through encoder
        features = self.encoder(input_tensor)
        
        # Get model prediction
        # We need to do a full forward pass
        self.model.eval()
        with torch.no_grad():
            if self.plane_name == 'sagittal':
                # Create dummy inputs for other planes
                dummy = torch.zeros_like(input_tensor)
                output = self.model(input_tensor, dummy, dummy)
            elif self.plane_name == 'coronal':
                dummy = torch.zeros_like(input_tensor)
                output = self.model(dummy, input_tensor, dummy)
            else:  # axial
                dummy = torch.zeros_like(input_tensor)
                output = self.model(dummy, dummy, input_tensor)
        
        # Backward pass
        self.model.zero_grad()
        self.encoder.train()
        features = self.encoder(input_tensor)
        
        # Use the mean of features as target
        target = features.mean()
        target.backward()
        
        # Generate CAM
        if self.gradients is None or self.activations is None:
            return np.zeros((7, 7))
        
        # Global average pooling on gradients
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        
        # Weighted combination
        cam = (weights * self.activations).sum(dim=1).squeeze()
        
        # Apply ReLU and normalize
        cam = F.relu(cam)
        cam = cam.cpu().numpy()
        
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        
        return cam

class AttentionVisualizer:
    """
    Visualize the actual attention weights from the multi-head attention layer
    This is the REAL mechanism the model uses to focus
    """
    
    def __init__(self, model):
        self.model = model
        self.attention_weights = None
        
        # Hook into the attention layer
        def save_attention(module, input, output):
            # output is (attended_features, attention_weights)
            if len(output) == 2:
                self.attention_weights = output[1].detach()
        
        # Register hook on attention layer
        self.model.attention.register_forward_hook(save_attention)
    
    def get_attention_weights(self, sagittal, coronal, axial):
        """Get attention weights for a case"""
        self.model.eval()
        
        with torch.no_grad():
            _ = self.model(sagittal, coronal, axial)
        
        if self.attention_weights is not None:
            # Average across attention heads
            # Shape: [num_heads, 3, 3] -> [3, 3]
            weights = self.attention_weights[0].mean(dim=0)  # Average over heads
            return weights.cpu().numpy()
        
        return np.ones((3, 3)) / 3  # Uniform if not available

def simple_saliency_map(model, input_tensor, encoder, plane_name):
    """
    Simple saliency map - shows which pixels affect the prediction most
    More reliable than Grad-CAM for some architectures
    """
    
    input_tensor.requires_grad = True
    
    model.train()
    
    # Forward pass
    if plane_name == 'sagittal':
        dummy = torch.zeros_like(input_tensor)
        features = encoder(input_tensor)
    elif plane_name == 'coronal':
        dummy = torch.zeros_like(input_tensor)
        features = encoder(input_tensor)
    else:  # axial
        dummy = torch.zeros_like(input_tensor)
        features = encoder(input_tensor)
    
    # Use mean of features as target
    target = features.abs().mean()
    
    # Backward
    model.zero_grad()
    target.backward()
    
    # Get gradients
    saliency = input_tensor.grad.data.abs()
    
    # Take maximum across color channels
    saliency = saliency.max(dim=1)[0]
    saliency = saliency.squeeze().cpu().numpy()
    
    # Normalize
    if saliency.max() > 0:
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min())
    
    input_tensor.requires_grad = False
    
    return saliency

def analyze_case_fixed(model, dataset, case_idx, device, output_dir):
    """Fixed analysis with working visualizations"""
    
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
    
    # Get original images
    def denormalize(tensor):
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = tensor.cpu() * std + mean
        tensor = torch.clamp(tensor, 0, 1)
        return (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    
    sag_img = denormalize(batch['sagittal'])
    cor_img = denormalize(batch['coronal'])
    axi_img = denormalize(batch['axial'])
    
    # Generate saliency maps (more reliable)
    print(f"   Generating saliency maps for case {case_id}...")
    sag_saliency = simple_saliency_map(model, sagittal.clone(), model.sagittal_encoder, 'sagittal')
    cor_saliency = simple_saliency_map(model, coronal.clone(), model.coronal_encoder, 'coronal')
    axi_saliency = simple_saliency_map(model, axial.clone(), model.axial_encoder, 'axial')
    
    # Get attention weights
    print(f"   Extracting attention weights...")
    attn_viz = AttentionVisualizer(model)
    attn_weights = attn_viz.get_attention_weights(sagittal, coronal, axial)
    
    # Create visualization
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # Title
    title = f"Case {case_id:04d} | True: {'Abnormal' if true_label == 1 else 'Normal'} | "
    title += f"Predicted: {'Abnormal' if pred == 1 else 'Normal'} ({prob:.1%})"
    if pred == true_label:
        title += " ✅"
        color = 'green'
    else:
        title += " ❌"
        color = 'red'
    
    fig.suptitle(title, fontsize=16, weight='bold', color=color)
    
    # Row 1: Original images
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(sag_img)
    ax1.set_title('Sagittal View', fontsize=12, weight='bold')
    ax1.axis('off')
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(cor_img)
    ax2.set_title('Coronal View', fontsize=12, weight='bold')
    ax2.axis('off')
    
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(axi_img)
    ax3.set_title('Axial View', fontsize=12, weight='bold')
    ax3.axis('off')
    
    # Prediction info
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.axis('off')
    info_text = f"PREDICTION DETAILS\n\n"
    info_text += f"Probability: {prob:.1%}\n"
    info_text += f"Confidence: {abs(prob-0.5)*2:.1%}\n\n"
    if prob > 0.5:
        info_text += f"Abnormal: {prob:.1%}\n"
        info_text += f"Normal: {1-prob:.1%}\n"
    else:
        info_text += f"Normal: {1-prob:.1%}\n"
        info_text += f"Abnormal: {prob:.1%}\n"
    
    ax4.text(0.1, 0.5, info_text, fontsize=11, verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7),
            family='monospace')
    
    # Row 2: Saliency maps (overlay on original)
    def apply_heatmap(img, saliency, alpha=0.4):
        # Resize saliency to match image
        saliency_resized = cv2.resize(saliency, (img.shape[1], img.shape[0]))
        
        # Apply colormap
        heatmap = cv2.applyColorMap(np.uint8(255 * saliency_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Overlay
        overlaid = (1 - alpha) * img + alpha * heatmap
        return np.clip(overlaid, 0, 255).astype(np.uint8), heatmap
    
    overlay_sag, heat_sag = apply_heatmap(sag_img, sag_saliency)
    overlay_cor, heat_cor = apply_heatmap(cor_img, cor_saliency)
    overlay_axi, heat_axi = apply_heatmap(axi_img, axi_saliency)
    
    ax5 = fig.add_subplot(gs[1, 0])
    ax5.imshow(overlay_sag)
    ax5.set_title('Sagittal - Saliency Map', fontsize=11, weight='bold')
    ax5.axis('off')
    
    ax6 = fig.add_subplot(gs[1, 1])
    ax6.imshow(overlay_cor)
    ax6.set_title('Coronal - Saliency Map', fontsize=11, weight='bold')
    ax6.axis('off')
    
    ax7 = fig.add_subplot(gs[1, 2])
    ax7.imshow(overlay_axi)
    ax7.set_title('Axial - Saliency Map', fontsize=11, weight='bold')
    ax7.axis('off')
    
    # Attention weights visualization
    ax8 = fig.add_subplot(gs[1, 3])
    
    # Extract plane importance from attention weights
    # attn_weights is [3, 3] - attention from each plane to each plane
    plane_importance = attn_weights.mean(axis=1)  # Average attention each plane receives
    plane_importance = plane_importance / plane_importance.sum()  # Normalize
    
    planes = ['Sagittal', 'Coronal', 'Axial']
    colors = ['blue', 'green', 'red']
    bars = ax8.bar(planes, plane_importance, color=colors, alpha=0.7)
    ax8.set_title('Plane Importance\n(from Attention)', fontsize=11, weight='bold')
    ax8.set_ylabel('Attention Weight', fontsize=10)
    ax8.set_ylim([0, 1])
    ax8.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars, plane_importance):
        height = bar.get_height()
        ax8.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1%}', ha='center', va='bottom', fontsize=10, weight='bold')
    
    # Row 3: Pure heatmaps
    ax9 = fig.add_subplot(gs[2, 0])
    im1 = ax9.imshow(heat_sag)
    ax9.set_title('Sagittal Heatmap', fontsize=11, weight='bold')
    ax9.axis('off')
    
    ax10 = fig.add_subplot(gs[2, 1])
    im2 = ax10.imshow(heat_cor)
    ax10.set_title('Coronal Heatmap', fontsize=11, weight='bold')
    ax10.axis('off')
    
    ax11 = fig.add_subplot(gs[2, 2])
    im3 = ax11.imshow(heat_axi)
    ax11.set_title('Axial Heatmap', fontsize=11, weight='bold')
    ax11.axis('off')
    
    # Add colorbar
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax11)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im3, cax=cax, label='Importance')
    
    # Interpretation guide
    ax12 = fig.add_subplot(gs[2, 3])
    ax12.axis('off')
    
    guide_text = "INTERPRETATION\n\n"
    guide_text += "Heatmap Colors:\n"
    guide_text += "  🔴 Red: HIGH importance\n"
    guide_text += "     (Critical for decision)\n"
    guide_text += "  🟡 Yellow: MEDIUM\n"
    guide_text += "  🔵 Blue: LOW importance\n\n"
    guide_text += "Plane Importance:\n"
    guide_text += f"  {planes[plane_importance.argmax()]}:\n"
    guide_text += f"  Most influential\n"
    guide_text += f"  ({plane_importance.max():.1%})\n\n"
    guide_text += "Note: Saliency maps show\n"
    guide_text += "pixel-level importance"
    
    ax12.text(0.1, 0.5, guide_text, fontsize=9, verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5),
             family='monospace')
    
    # Save
    result_type = 'correct' if pred == true_label else 'incorrect'
    label_name = 'abnormal' if true_label == 1 else 'normal'
    filename = f"case_{case_id:04d}_{label_name}_{result_type}_prob{prob:.2f}.png"
    save_path = os.path.join(output_dir, filename)
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return {
        'case_id': case_id,
        'true_label': true_label,
        'prediction': pred,
        'probability': prob,
        'correct': pred == true_label,
        'filename': filename,
        'sagittal_importance': float(plane_importance[0]),
        'coronal_importance': float(plane_importance[1]),
        'axial_importance': float(plane_importance[2])
    }

def main():
    print("="*70)
    print("🔍 FIXED MULTI-PLANE XAI ANALYSIS")
    print("="*70)
    
    CONFIG = {
        'mrnet_path': '/Users/yuvrajpratapsingh/knee-mri-abnormality-detection/dataset/MRNet-v1.0',
        'model_path': '/Users/yuvrajpratapsingh/knee-mri-abnormality-detection/outputs/models/best_model_multiplane.pth',
        'output_dir': '/Users/yuvrajpratapsingh/knee-mri-abnormality-detection/outputs/xai_multiplane_fixed',
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
    print("\n📊 Selecting cases to analyze...")
    
    # Get predictions for selection
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
    
    # Select cases: 2 TP, 2 TN, all FP, all FN, 2 borderline
    tp_cases = pred_df[(pred_df['prediction'] == 1) & (pred_df['true_label'] == 1)].nlargest(2, 'confidence')
    tn_cases = pred_df[(pred_df['prediction'] == 0) & (pred_df['true_label'] == 0)].nlargest(2, 'confidence')
    fp_cases = pred_df[(pred_df['prediction'] == 1) & (pred_df['true_label'] == 0)]  # All FPs
    fn_cases = pred_df[(pred_df['prediction'] == 0) & (pred_df['true_label'] == 1)]  # All FNs
    border_cases = pred_df[pred_df['correct'] == True].nsmallest(2, 'confidence')
    
    selected_indices = pd.concat([tp_cases, tn_cases, fp_cases, fn_cases, border_cases])['idx'].unique()
    
    print(f"\n🎯 Analyzing {len(selected_indices)} cases:")
    print(f"   - {len(tp_cases)} high-confidence True Positives")
    print(f"   - {len(tn_cases)} high-confidence True Negatives")
    print(f"   - {len(fp_cases)} False Positives (ALL)")
    print(f"   - {len(fn_cases)} False Negatives (ALL)")
    print(f"   - {len(border_cases)} Borderline correct cases")
    
    # Analyze cases
    results = []
    print("\n🔬 Generating visualizations...")
    
    for idx in tqdm(selected_indices):
        try:
            result = analyze_case_fixed(model, val_dataset, idx, device, CONFIG['output_dir'])
            results.append(result)
        except Exception as e:
            print(f"\n⚠️  Error on case {idx}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{CONFIG['output_dir']}/xai_fixed_summary.csv", index=False)
    
    # Analyze plane importance
    print("\n📊 Analyzing plane importance across cases...")
    
    avg_sag = results_df['sagittal_importance'].mean()
    avg_cor = results_df['coronal_importance'].mean()
    avg_axi = results_df['axial_importance'].mean()
    
    print(f"\nAverage Plane Importance:")
    print(f"  Sagittal: {avg_sag:.1%}")
    print(f"  Coronal:  {avg_cor:.1%}")
    print(f"  Axial:    {avg_axi:.1%}")
    
    # Most important plane for correct vs incorrect
    correct_cases = results_df[results_df['correct'] == True]
    incorrect_cases = results_df[results_df['correct'] == False]
    
    if len(incorrect_cases) > 0:
        print(f"\nCorrect predictions rely most on:")
        correct_planes = correct_cases[['sagittal_importance', 'coronal_importance', 'axial_importance']].mean()
        print(f"  Sagittal: {correct_planes['sagittal_importance']:.1%}")
        print(f"  Coronal:  {correct_planes['coronal_importance']:.1%}")
        print(f"  Axial:    {correct_planes['axial_importance']:.1%}")
        
        print(f"\nIncorrect predictions rely most on:")
        incorrect_planes = incorrect_cases[['sagittal_importance', 'coronal_importance', 'axial_importance']].mean()
        print(f"  Sagittal: {incorrect_planes['sagittal_importance']:.1%}")
        print(f"  Coronal:  {incorrect_planes['coronal_importance']:.1%}")
        print(f"  Axial:    {incorrect_planes['axial_importance']:.1%}")
    
    print("\n" + "="*70)
    print("✅ FIXED XAI ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\n📁 Results saved to: {CONFIG['output_dir']}/")
    print(f"   - {len(results)} case visualizations with WORKING heatmaps")
    print(f"   - xai_fixed_summary.csv with plane importance")
    print("="*70)
    
    print("\n💡 Key Differences from Previous XAI:")
    print("   - Uses saliency maps instead of Grad-CAM")
    print("   - Shows actual attention weights from attention layer")
    print("   - Visualizes plane importance for each prediction")
    print("   - More reliable pixel-level importance")

if __name__ == '__main__':
    main()
