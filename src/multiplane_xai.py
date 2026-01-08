import os
import sys
import torch
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

class GradCAM:
    """Grad-CAM for visualizing what the model focuses on"""
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_tensor, target_class=None):
        # Forward pass
        output = self.model.forward(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1)
        
        # Backward pass
        self.model.zero_grad()
        output[0, target_class].backward(retain_graph=True)
        
        # Generate CAM
        gradients = self.gradients[0]
        activations = self.activations[0]
        
        # Global average pooling of gradients
        weights = gradients.mean(dim=(1, 2), keepdim=True)
        
        # Weighted combination of activation maps
        cam = (weights * activations).sum(dim=0)
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam.cpu().numpy()

def apply_colormap_on_image(org_img, cam, alpha=0.5):
    """Overlay CAM heatmap on original image"""
    # Resize CAM to match image size
    cam_resized = cv2.resize(cam, (org_img.shape[1], org_img.shape[0]))
    
    # Apply colormap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Overlay
    overlaid = (1 - alpha) * org_img + alpha * heatmap
    overlaid = np.clip(overlaid, 0, 255).astype(np.uint8)
    
    return overlaid, heatmap

def analyze_case_with_xai(model, dataset, case_idx, device, output_dir):
    """Analyze a single case with XAI visualizations"""
    
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
    
    # Get original images (denormalize)
    def denormalize(tensor):
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = tensor.cpu() * std + mean
        tensor = torch.clamp(tensor, 0, 1)
        return (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    
    sag_img = denormalize(batch['sagittal'])
    cor_img = denormalize(batch['coronal'])
    axi_img = denormalize(batch['axial'])
    
    # Create figure
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # Title with prediction info
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
    info_text += f"Classification:\n"
    if prob > 0.5:
        info_text += f"  Abnormal: {prob:.1%}\n"
        info_text += f"  Normal: {1-prob:.1%}\n"
    else:
        info_text += f"  Normal: {1-prob:.1%}\n"
        info_text += f"  Abnormal: {prob:.1%}\n"
    
    info_text += f"\nDecision: "
    info_text += "High Confidence" if abs(prob-0.5) > 0.3 else "Low Confidence"
    
    ax4.text(0.1, 0.5, info_text, fontsize=11, verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7),
            family='monospace')
    
    # Row 2: Grad-CAM heatmaps
    try:
        # Create temporary model for Grad-CAM
        model_gradcam = MultiPlaneFusion(num_classes=1, dropout_rate=0.4).to(device)
        model_gradcam.load_state_dict(model.state_dict())
        model_gradcam.eval()
        
        # Get attention for each plane
        # Sagittal
        gradcam_sag = GradCAM(model_gradcam.sagittal_encoder, 
                              model_gradcam.sagittal_encoder[-1])
        cam_sag = gradcam_sag.generate_cam(sagittal)
        overlay_sag, heat_sag = apply_colormap_on_image(sag_img, cam_sag)
        
        ax5 = fig.add_subplot(gs[1, 0])
        ax5.imshow(overlay_sag)
        ax5.set_title('Sagittal - Attention Map', fontsize=11, weight='bold')
        ax5.axis('off')
        
        # Coronal
        gradcam_cor = GradCAM(model_gradcam.coronal_encoder, 
                              model_gradcam.coronal_encoder[-1])
        cam_cor = gradcam_cor.generate_cam(coronal)
        overlay_cor, heat_cor = apply_colormap_on_image(cor_img, cam_cor)
        
        ax6 = fig.add_subplot(gs[1, 1])
        ax6.imshow(overlay_cor)
        ax6.set_title('Coronal - Attention Map', fontsize=11, weight='bold')
        ax6.axis('off')
        
        # Axial
        gradcam_axi = GradCAM(model_gradcam.axial_encoder, 
                              model_gradcam.axial_encoder[-1])
        cam_axi = gradcam_axi.generate_cam(axial)
        overlay_axi, heat_axi = apply_colormap_on_image(axi_img, cam_axi)
        
        ax7 = fig.add_subplot(gs[1, 2])
        ax7.imshow(overlay_axi)
        ax7.set_title('Axial - Attention Map', fontsize=11, weight='bold')
        ax7.axis('off')
        
    except Exception as e:
        print(f"⚠️  Grad-CAM generation failed: {e}")
        print("   Continuing with basic visualizations...")
    
    # Attention weight comparison
    ax8 = fig.add_subplot(gs[1, 3])
    planes = ['Sagittal', 'Coronal', 'Axial']
    weights = [0.33, 0.33, 0.34]  # Placeholder - would need actual attention weights
    ax8.bar(planes, weights, color=['blue', 'green', 'red'], alpha=0.7)
    ax8.set_title('Plane Contribution', fontsize=11, weight='bold')
    ax8.set_ylabel('Attention Weight', fontsize=10)
    ax8.set_ylim([0, 1])
    ax8.grid(axis='y', alpha=0.3)
    
    # Row 3: Pure heatmaps
    ax9 = fig.add_subplot(gs[2, 0])
    try:
        ax9.imshow(heat_sag)
        ax9.set_title('Sagittal Heatmap', fontsize=11, weight='bold')
    except:
        ax9.text(0.5, 0.5, 'N/A', ha='center', va='center')
    ax9.axis('off')
    
    ax10 = fig.add_subplot(gs[2, 1])
    try:
        ax10.imshow(heat_cor)
        ax10.set_title('Coronal Heatmap', fontsize=11, weight='bold')
    except:
        ax10.text(0.5, 0.5, 'N/A', ha='center', va='center')
    ax10.axis('off')
    
    ax11 = fig.add_subplot(gs[2, 2])
    try:
        ax11.imshow(heat_axi)
        ax11.set_title('Axial Heatmap', fontsize=11, weight='bold')
    except:
        ax11.text(0.5, 0.5, 'N/A', ha='center', va='center')
    ax11.axis('off')
    
    # Interpretation guide
    ax12 = fig.add_subplot(gs[2, 3])
    ax12.axis('off')
    guide_text = "INTERPRETATION GUIDE\n\n"
    guide_text += "Heatmap Colors:\n"
    guide_text += "  🔴 Red: High attention\n"
    guide_text += "     (Model focuses here)\n"
    guide_text += "  🟡 Yellow: Medium attention\n"
    guide_text += "  🔵 Blue: Low attention\n\n"
    guide_text += "What to look for:\n"
    guide_text += "• Red areas show regions\n"
    guide_text += "  influencing the decision\n"
    guide_text += "• Multiple planes provide\n"
    guide_text += "  complementary information\n"
    guide_text += "• High confidence = strong\n"
    guide_text += "  consistent signals"
    
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
        'filename': filename
    }

def main():
    print("="*70)
    print("🔍 MULTI-PLANE MODEL XAI ANALYSIS")
    print("="*70)
    
    CONFIG = {
        'mrnet_path': '/Users/yuvrajpratapsingh/knee-mri-abnormality-detection/dataset/MRNet-v1.0',
        'model_path': '/Users/yuvrajpratapsingh/knee-mri-abnormality-detection/outputs/models/best_model_multiplane.pth',
        'output_dir': '/Users/yuvrajpratapsingh/knee-mri-abnormality-detection/outputs/xai_multiplane',
        'task': 'abnormal',
        'num_examples': 20,  # Number of cases to analyze
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
    model.eval()
    
    print(f"✅ Model loaded from epoch {checkpoint['epoch']+1}")
    
    # Get predictions for all cases
    print("\n🔍 Getting predictions for all cases...")
    all_predictions = []
    
    with torch.no_grad():
        for idx in tqdm(range(len(val_dataset))):
            batch = val_dataset[idx]
            sagittal = batch['sagittal'].unsqueeze(0).to(device)
            coronal = batch['coronal'].unsqueeze(0).to(device)
            axial = batch['axial'].unsqueeze(0).to(device)
            label = batch['label'].item()
            
            output = model(sagittal, coronal, axial)
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
    
    # Select interesting cases
    print("\n📊 Selecting interesting cases...")
    
    # Categories
    tp_cases = pred_df[(pred_df['prediction'] == 1) & (pred_df['true_label'] == 1)].nlargest(3, 'confidence')
    tn_cases = pred_df[(pred_df['prediction'] == 0) & (pred_df['true_label'] == 0)].nlargest(3, 'confidence')
    fp_cases = pred_df[(pred_df['prediction'] == 1) & (pred_df['true_label'] == 0)].nlargest(2, 'confidence')
    fn_cases = pred_df[(pred_df['prediction'] == 0) & (pred_df['true_label'] == 1)].nlargest(2, 'confidence')
    
    # Edge cases (low confidence)
    edge_correct = pred_df[pred_df['correct'] == True].nsmallest(3, 'confidence')
    edge_incorrect = pred_df[pred_df['correct'] == False].nsmallest(2, 'confidence')
    
    selected_indices = pd.concat([
        tp_cases, tn_cases, fp_cases, fn_cases, edge_correct, edge_incorrect
    ])['idx'].unique()
    
    print(f"\n🎯 Analyzing {len(selected_indices)} selected cases:")
    print(f"   - {len(tp_cases)} True Positives (high confidence)")
    print(f"   - {len(tn_cases)} True Negatives (high confidence)")
    print(f"   - {len(fp_cases)} False Positives")
    print(f"   - {len(fn_cases)} False Negatives")
    print(f"   - {len(edge_correct)} Edge cases (correct)")
    print(f"   - {len(edge_incorrect)} Edge cases (incorrect)")
    
    # Analyze selected cases
    results = []
    print("\n🔬 Generating XAI visualizations...")
    
    for idx in tqdm(selected_indices):
        try:
            result = analyze_case_with_xai(model, val_dataset, idx, device, CONFIG['output_dir'])
            results.append(result)
        except Exception as e:
            print(f"\n⚠️  Error analyzing case {idx}: {e}")
            continue
    
    # Save summary
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{CONFIG['output_dir']}/xai_analysis_summary.csv", index=False)
    
    # Create summary visualization
    create_summary_visualization(results_df, pred_df, CONFIG['output_dir'])
    
    print("\n" + "="*70)
    print("✅ XAI ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\n📁 Results saved to: {CONFIG['output_dir']}/")
    print(f"   - {len(results)} case visualizations")
    print(f"   - xai_analysis_summary.csv")
    print(f"   - xai_summary_stats.png")
    print("="*70)

def create_summary_visualization(results_df, all_predictions_df, output_dir):
    """Create summary statistics visualization"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Correct vs Incorrect distribution
    correct_counts = results_df['correct'].value_counts()
    axes[0, 0].pie(correct_counts, labels=['Incorrect', 'Correct'], autopct='%1.1f%%',
                   colors=['lightcoral', 'lightgreen'], startangle=90)
    axes[0, 0].set_title('Analyzed Cases: Correct vs Incorrect', fontsize=12, weight='bold')
    
    # 2. Probability distribution
    axes[0, 1].hist(all_predictions_df['probability'], bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(0.5, color='red', linestyle='--', linewidth=2, label='Threshold')
    axes[0, 1].set_xlabel('Predicted Probability', fontsize=11)
    axes[0, 1].set_ylabel('Count', fontsize=11)
    axes[0, 1].set_title('Prediction Probability Distribution', fontsize=12, weight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # 3. Confidence distribution
    axes[0, 2].hist(all_predictions_df['confidence'], bins=20, color='lightgreen', edgecolor='black', alpha=0.7)
    axes[0, 2].set_xlabel('Confidence', fontsize=11)
    axes[0, 2].set_ylabel('Count', fontsize=11)
    axes[0, 2].set_title('Prediction Confidence Distribution', fontsize=12, weight='bold')
    axes[0, 2].grid(alpha=0.3)
    
    # 4. Confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(all_predictions_df['true_label'], all_predictions_df['prediction'])
    
    import seaborn as sns
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0],
                xticklabels=['Normal', 'Abnormal'],
                yticklabels=['Normal', 'Abnormal'],
                cbar_kws={'label': 'Count'})
    axes[1, 0].set_xlabel('Predicted', fontsize=11, weight='bold')
    axes[1, 0].set_ylabel('True', fontsize=11, weight='bold')
    axes[1, 0].set_title('Confusion Matrix', fontsize=12, weight='bold')
    
    # 5. Performance by confidence level
    bins = pd.cut(all_predictions_df['confidence'], bins=5)
    accuracy_by_conf = all_predictions_df.groupby(bins)['correct'].mean()
    
    axes[1, 1].plot(range(len(accuracy_by_conf)), accuracy_by_conf.values, 'o-', linewidth=2, markersize=8)
    axes[1, 1].set_xlabel('Confidence Level', fontsize=11)
    axes[1, 1].set_ylabel('Accuracy', fontsize=11)
    axes[1, 1].set_title('Accuracy vs Confidence', fontsize=12, weight='bold')
    axes[1, 1].set_ylim([0, 1])
    axes[1, 1].grid(alpha=0.3)
    axes[1, 1].set_xticklabels(['Low', '', 'Med', '', 'High'])
    
    # 6. Summary statistics
    axes[1, 2].axis('off')
    
    tp = len(all_predictions_df[(all_predictions_df['prediction'] == 1) & (all_predictions_df['true_label'] == 1)])
    tn = len(all_predictions_df[(all_predictions_df['prediction'] == 0) & (all_predictions_df['true_label'] == 0)])
    fp = len(all_predictions_df[(all_predictions_df['prediction'] == 1) & (all_predictions_df['true_label'] == 0)])
    fn = len(all_predictions_df[(all_predictions_df['prediction'] == 0) & (all_predictions_df['true_label'] == 1)])
    
    summary_text = "SUMMARY STATISTICS\n\n"
    summary_text += f"Total Cases: {len(all_predictions_df)}\n\n"
    summary_text += f"True Positives:  {tp}\n"
    summary_text += f"True Negatives:  {tn}\n"
    summary_text += f"False Positives: {fp}\n"
    summary_text += f"False Negatives: {fn}\n\n"
    summary_text += f"Sensitivity: {tp/(tp+fn):.1%}\n"
    summary_text += f"Specificity: {tn/(tn+fp):.1%}\n"
    summary_text += f"Accuracy: {(tp+tn)/len(all_predictions_df):.1%}\n\n"
    summary_text += f"Avg Confidence:\n"
    summary_text += f"  Correct: {all_predictions_df[all_predictions_df['correct']]['confidence'].mean():.1%}\n"
    summary_text += f"  Incorrect: {all_predictions_df[~all_predictions_df['correct']]['confidence'].mean():.1%}"
    
    axes[1, 2].text(0.1, 0.5, summary_text, fontsize=11, verticalalignment='center',
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7),
                   family='monospace')
    
    plt.suptitle('XAI ANALYSIS SUMMARY', fontsize=16, weight='bold')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/xai_summary_stats.png", dpi=200, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    main()
