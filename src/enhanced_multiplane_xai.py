import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
import cv2
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.multiplane_loader import MultiPlaneMRNetDataset
from src.multiplane_model import MultiPlaneFusion

class EnhancedSaliencyMap:
    """
    Enhanced saliency map with better visualization and region detection
    """
    
    def __init__(self, model, encoder, plane_name):
        self.model = model
        self.encoder = encoder
        self.plane_name = plane_name
    
    def generate_saliency(self, input_tensor):
        """Generate high-quality saliency map"""
        
        input_tensor = input_tensor.clone().detach().requires_grad_(True)
        
        self.model.train()
        self.encoder.train()
        
        # Forward pass
        features = self.encoder(input_tensor)
        
        # Use absolute mean as target (shows what model focuses on)
        target = features.abs().mean()
        
        # Backward
        self.model.zero_grad()
        if input_tensor.grad is not None:
            input_tensor.grad.zero_()
        
        target.backward()
        
        # Get gradients
        saliency = input_tensor.grad.data.abs()
        
        # Take maximum across color channels
        saliency = saliency.max(dim=1)[0]
        saliency = saliency.squeeze().cpu().numpy()
        
        # Apply Gaussian smoothing for better visualization
        from scipy.ndimage import gaussian_filter
        saliency = gaussian_filter(saliency, sigma=2)
        
        # Normalize
        if saliency.max() > 0:
            saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min())
        
        return saliency

class AnatomicalRegionAnalyzer:
    """
    Analyze which anatomical regions are most important
    """
    
    # Define anatomical regions for each plane
    REGIONS = {
        'sagittal': {
            'ACL': {'y': (0.3, 0.6), 'x': (0.4, 0.7), 'name': 'ACL Region'},
            'PCL': {'y': (0.4, 0.7), 'x': (0.3, 0.6), 'name': 'PCL Region'},
            'Meniscus': {'y': (0.35, 0.55), 'x': (0.45, 0.75), 'name': 'Meniscus'},
            'Cartilage': {'y': (0.3, 0.5), 'x': (0.5, 0.8), 'name': 'Cartilage'},
        },
        'coronal': {
            'Medial_Meniscus': {'y': (0.4, 0.6), 'x': (0.3, 0.5), 'name': 'Medial Meniscus'},
            'Lateral_Meniscus': {'y': (0.4, 0.6), 'x': (0.5, 0.7), 'name': 'Lateral Meniscus'},
            'Joint_Space': {'y': (0.45, 0.55), 'x': (0.3, 0.7), 'name': 'Joint Space'},
        },
        'axial': {
            'Patella': {'y': (0.2, 0.5), 'x': (0.3, 0.7), 'name': 'Patella'},
            'Femur': {'y': (0.5, 0.8), 'x': (0.3, 0.7), 'name': 'Femur'},
            'Soft_Tissue': {'y': (0.3, 0.7), 'x': (0.2, 0.8), 'name': 'Soft Tissue'},
        }
    }
    
    @staticmethod
    def analyze_regions(saliency_map, plane_name):
        """Identify which anatomical regions show high importance"""
        
        h, w = saliency_map.shape
        regions = AnatomicalRegionAnalyzer.REGIONS.get(plane_name, {})
        
        region_scores = {}
        
        for region_key, region_def in regions.items():
            # Get region bounds
            y_start = int(region_def['y'][0] * h)
            y_end = int(region_def['y'][1] * h)
            x_start = int(region_def['x'][0] * w)
            x_end = int(region_def['x'][1] * w)
            
            # Extract region
            region_map = saliency_map[y_start:y_end, x_start:x_end]
            
            # Calculate importance score
            score = region_map.mean()
            
            region_scores[region_def['name']] = {
                'score': float(score),
                'bounds': (x_start, y_start, x_end - x_start, y_end - y_start)
            }
        
        return region_scores

def apply_enhanced_heatmap(img, saliency, alpha=0.5):
    """
    Apply enhanced heatmap with better color mapping
    """
    # Resize saliency to match image
    saliency_resized = cv2.resize(saliency, (img.shape[1], img.shape[0]))
    
    # Create custom colormap: blue (low) -> green -> yellow -> red (high)
    from matplotlib import cm
    colormap = cm.get_cmap('jet')
    
    # Apply colormap
    heatmap_colored = colormap(saliency_resized)[:, :, :3]  # RGB only
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
    
    # Overlay
    overlaid = (1 - alpha) * img + alpha * heatmap_colored
    overlaid = np.clip(overlaid, 0, 255).astype(np.uint8)
    
    return overlaid, heatmap_colored

def create_enhanced_legend(ax, include_regions=True):
    """
    Create comprehensive legend explaining the visualization
    """
    
    legend_text = "🔍 INTERPRETATION GUIDE\n\n"
    legend_text += "HEATMAP COLORS:\n"
    legend_text += "🔴 RED: Critical importance\n"
    legend_text += "   (Primary focus area)\n"
    legend_text += "   Model strongly relies on\n"
    legend_text += "   these regions\n\n"
    legend_text += "🟡 YELLOW: High importance\n"
    legend_text += "   (Secondary focus)\n\n"
    legend_text += "🟢 GREEN: Medium importance\n"
    legend_text += "   (Supporting evidence)\n\n"
    legend_text += "🔵 BLUE: Low importance\n"
    legend_text += "   (Minimal influence)\n\n"
    
    if include_regions:
        legend_text += "ANATOMICAL REGIONS:\n"
        legend_text += "• Boxes show key structures\n"
        legend_text += "• Scores indicate importance\n"
        legend_text += "• Higher = more influential\n"
    
    ax.text(0.05, 0.95, legend_text, 
           transform=ax.transAxes,
           fontsize=9,
           verticalalignment='top',
           bbox=dict(boxstyle='round', 
                    facecolor='white', 
                    edgecolor='black',
                    alpha=0.9,
                    linewidth=2),
           family='monospace')

def analyze_case_enhanced(model, dataset, case_idx, device, output_dir):
    """
    Enhanced case analysis with better explanations and visualizations
    """
    
    # Get case data
    batch = dataset[case_idx]
    case_id = dataset.labels_df.iloc[case_idx]['case']
    true_label = batch['label'].item()
    
    sagittal = batch['sagittal'].unsqueeze(0).to(device)
    coronal = batch['coronal'].unsqueeze(0).to(device)
    axial = batch['axial'].unsqueeze(0).to(device)
    
    # Get prediction with temperature scaling for better calibration
    model.eval()
    with torch.no_grad():
        output = model(sagittal, coronal, axial)
        
        # Apply temperature scaling (you can tune this)
        temperature =  1.4069 # From your evaluation
        calibrated_output = output / temperature
        
        raw_prob = torch.sigmoid(output).item()
        calibrated_prob = torch.sigmoid(calibrated_output).item()
        pred = 1 if calibrated_prob > 0.5 else 0
    
    # Calculate uncertainty (entropy)
    p = calibrated_prob
    uncertainty = -p * np.log(p + 1e-10) - (1-p) * np.log(1-p + 1e-10)
    confidence = abs(calibrated_prob - 0.5) * 2
    
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
    
    # Generate saliency maps
    print(f"   Generating enhanced saliency maps for case {case_id}...")
    
    sag_mapper = EnhancedSaliencyMap(model, model.sagittal_encoder, 'sagittal')
    cor_mapper = EnhancedSaliencyMap(model, model.coronal_encoder, 'coronal')
    axi_mapper = EnhancedSaliencyMap(model, model.axial_encoder, 'axial')
    
    sag_saliency = sag_mapper.generate_saliency(sagittal.clone())
    cor_saliency = cor_mapper.generate_saliency(coronal.clone())
    axi_saliency = axi_mapper.generate_saliency(axial.clone())
    
    # Analyze anatomical regions
    sag_regions = AnatomicalRegionAnalyzer.analyze_regions(sag_saliency, 'sagittal')
    cor_regions = AnatomicalRegionAnalyzer.analyze_regions(cor_saliency, 'coronal')
    axi_regions = AnatomicalRegionAnalyzer.analyze_regions(axi_saliency, 'axial')
    
    # Create enhanced visualization
    fig = plt.figure(figsize=(24, 14))
    gs = fig.add_gridspec(4, 4, hspace=0.35, wspace=0.3)
    
    # Main title with status
    title = f"Case {case_id:04d} | Ground Truth: {'ABNORMAL' if true_label == 1 else 'NORMAL'} | "
    title += f"Model Prediction: {'ABNORMAL' if pred == 1 else 'NORMAL'}"
    
    if pred == true_label:
        title += " ✅ CORRECT"
        title_color = 'darkgreen'
    else:
        title += " ❌ INCORRECT"
        title_color = 'darkred'
    
    fig.suptitle(title, fontsize=18, weight='bold', color=title_color, y=0.98)
    
    # Subtitle with probabilities
    subtitle = f"Raw Probability: {raw_prob:.1%} | Calibrated Probability: {calibrated_prob:.1%} | "
    subtitle += f"Confidence: {confidence:.1%} | Uncertainty: {uncertainty:.3f}"
    fig.text(0.5, 0.94, subtitle, ha='center', fontsize=12, style='italic')
    
    # Row 1: Original images with anatomical region boxes
    planes_data = [
        (sag_img, sag_saliency, sag_regions, 'Sagittal View', 0),
        (cor_img, cor_saliency, cor_regions, 'Coronal View', 1),
        (axi_img, axi_saliency, axi_regions, 'Axial View', 2)
    ]
    
    for img, saliency, regions, title, col in planes_data:
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(img)
        ax.set_title(title, fontsize=13, weight='bold')
        ax.axis('off')
        
        # Draw anatomical region boxes with highest importance
        if regions:
            top_region = max(regions.items(), key=lambda x: x[1]['score'])
            region_name, region_data = top_region
            x, y, w, h = region_data['bounds']
            score = region_data['score']
            
            # Draw box
            rect = Rectangle((x, y), w, h, 
                           linewidth=2, 
                           edgecolor='yellow', 
                           facecolor='none',
                           linestyle='--')
            ax.add_patch(rect)
            
            # Add label
            ax.text(x, y-5, f"{region_name}\n({score:.2f})", 
                   color='yellow', 
                   fontsize=9,
                   weight='bold',
                   bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    # Prediction details panel
    ax_info = fig.add_subplot(gs[0, 3])
    ax_info.axis('off')
    
    info_text = "📊 PREDICTION DETAILS\n\n"
    info_text += f"Ground Truth:\n"
    info_text += f"  {'🔴 ABNORMAL' if true_label == 1 else '🟢 NORMAL'}\n\n"
    info_text += f"Model Output:\n"
    info_text += f"  Raw: {raw_prob:.1%}\n"
    info_text += f"  Calibrated: {calibrated_prob:.1%}\n\n"
    
    if calibrated_prob > 0.5:
        info_text += f"Class Probabilities:\n"
        info_text += f"  Abnormal: {calibrated_prob:.1%}\n"
        info_text += f"  Normal: {1-calibrated_prob:.1%}\n\n"
    else:
        info_text += f"Class Probabilities:\n"
        info_text += f"  Normal: {1-calibrated_prob:.1%}\n"
        info_text += f"  Abnormal: {calibrated_prob:.1%}\n\n"
    
    info_text += f"Model Confidence: {confidence:.1%}\n"
    info_text += f"Uncertainty: {uncertainty:.3f}\n\n"
    
    if confidence < 0.3:
        info_text += "⚠️ LOW CONFIDENCE\n"
        info_text += "Model is uncertain\n"
        info_text += "Requires expert review"
    elif confidence < 0.6:
        info_text += "⚡ MEDIUM CONFIDENCE\n"
        info_text += "Borderline case"
    else:
        info_text += "✓ HIGH CONFIDENCE\n"
        info_text += "Model is certain"
    
    ax_info.text(0.05, 0.95, info_text, 
                transform=ax_info.transAxes,
                fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', 
                         facecolor='lightyellow', 
                         alpha=0.9,
                         linewidth=2),
                family='monospace')
    
    # Row 2: Saliency overlays
    for img, saliency, regions, title, col in planes_data:
        ax = fig.add_subplot(gs[1, col])
        overlay, _ = apply_enhanced_heatmap(img, saliency, alpha=0.5)
        ax.imshow(overlay)
        ax.set_title(f'{title.split()[0]} - Importance Overlay', 
                    fontsize=12, weight='bold')
        ax.axis('off')
    
    # Legend
    ax_legend = fig.add_subplot(gs[1, 3])
    ax_legend.axis('off')
    create_enhanced_legend(ax_legend, include_regions=True)
    
    # Row 3: Pure heatmaps with colorbars
    for img, saliency, regions, title, col in planes_data:
        ax = fig.add_subplot(gs[2, col])
        _, heatmap = apply_enhanced_heatmap(img, saliency, alpha=1.0)
        im = ax.imshow(heatmap)
        ax.set_title(f'{title.split()[0]} - Importance Map', 
                    fontsize=12, weight='bold')
        ax.axis('off')
        
        # Add colorbar
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label('Importance', rotation=270, labelpad=15)
    
    # Anatomical region importance panel
    ax_regions = fig.add_subplot(gs[2, 3])
    ax_regions.axis('off')
    
    region_text = "🎯 KEY FINDINGS\n\n"
    
    # Get top regions from each plane
    all_regions = []
    for plane, regions_dict in [('Sagittal', sag_regions), 
                                 ('Coronal', cor_regions), 
                                 ('Axial', axi_regions)]:
        if regions_dict:
            for region_name, data in regions_dict.items():
                all_regions.append((plane, region_name, data['score']))
    
    # Sort by importance
    all_regions.sort(key=lambda x: x[2], reverse=True)
    
    region_text += "Top Focus Areas:\n"
    for i, (plane, region, score) in enumerate(all_regions[:5], 1):
        if score > 0.5:
            marker = "🔴"
        elif score > 0.3:
            marker = "🟡"
        else:
            marker = "🟢"
        region_text += f"{i}. {marker} {region}\n"
        region_text += f"   ({plane}: {score:.2f})\n"
    
    ax_regions.text(0.05, 0.95, region_text,
                   transform=ax_regions.transAxes,
                   fontsize=10,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round',
                            facecolor='lightblue',
                            alpha=0.9,
                            linewidth=2),
                   family='monospace')
    
    # Row 4: Plane importance and clinical interpretation
    
    # Calculate plane importance from saliency
    plane_importance = np.array([
        sag_saliency.mean(),
        cor_saliency.mean(),
        axi_saliency.mean()
    ])
    plane_importance = plane_importance / plane_importance.sum()
    
    ax_planes = fig.add_subplot(gs[3, 0:2])
    planes = ['Sagittal', 'Coronal', 'Axial']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    bars = ax_planes.barh(planes, plane_importance, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    ax_planes.set_xlabel('Relative Importance', fontsize=12, weight='bold')
    ax_planes.set_title('Plane Contribution to Decision', fontsize=13, weight='bold')
    ax_planes.set_xlim([0, max(plane_importance) * 1.2])
    ax_planes.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Add value labels
    for bar, val in zip(bars, plane_importance):
        width = bar.get_width()
        ax_planes.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                      f'{val:.1%}',
                      ha='left', va='center',
                      fontsize=11, weight='bold')
    
    # Most important plane
    most_important_plane = planes[np.argmax(plane_importance)]
    ax_planes.text(0.02, 0.98, f"Primary: {most_important_plane}",
                  transform=ax_planes.transAxes,
                  fontsize=10, weight='bold',
                  va='top',
                  bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    # Clinical interpretation
    ax_clinical = fig.add_subplot(gs[3, 2:4])
    ax_clinical.axis('off')
    
    clinical_text = "🏥 CLINICAL INTERPRETATION\n\n"
    
    if pred == true_label:
        clinical_text += "✅ Model prediction matches\n"
        clinical_text += "   ground truth diagnosis\n\n"
    else:
        clinical_text += "⚠️ DISCREPANCY DETECTED\n"
        clinical_text += "   Requires expert review\n\n"
    
    # Decision explanation
    clinical_text += "Decision Basis:\n"
    if calibrated_prob > 0.7 or calibrated_prob < 0.3:
        clinical_text += f"• Strong {'abnormal' if calibrated_prob > 0.5 else 'normal'} pattern\n"
        clinical_text += f"• Detected in {most_important_plane} view\n"
    else:
        clinical_text += f"• Subtle/borderline findings\n"
        clinical_text += f"• Mixed signals across planes\n"
    
    # Top affected region
    if all_regions:
        top_plane, top_region, top_score = all_regions[0]
        clinical_text += f"• Key area: {top_region}\n"
        clinical_text += f"  ({top_plane} view)\n"
    
    clinical_text += f"\nRecommendation:\n"
    if confidence < 0.4:
        clinical_text += "⚠️ MANUAL REVIEW REQUIRED\n"
        clinical_text += "Model uncertainty is high\n"
        clinical_text += "Radiologist assessment needed"
    elif pred != true_label:
        clinical_text += "⚠️ VERIFY WITH RADIOLOGIST\n"
        clinical_text += "Model may have missed signs"
    else:
        clinical_text += "✓ Model confident in diagnosis\n"
        clinical_text += "Serves as screening assistance"
    
    ax_clinical.text(0.05, 0.95, clinical_text,
                    transform=ax_clinical.transAxes,
                    fontsize=10,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round',
                             facecolor='lightgreen' if pred == true_label else 'lightcoral',
                             alpha=0.8,
                             linewidth=2),
                    family='monospace')
    
    # Save
    result_type = 'correct' if pred == true_label else 'incorrect'
    label_name = 'abnormal' if true_label == 1 else 'normal'
    filename = f"enhanced_case_{case_id:04d}_{label_name}_{result_type}_prob{calibrated_prob:.2f}.png"
    save_path = os.path.join(output_dir, filename)
    
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return {
        'case_id': case_id,
        'true_label': true_label,
        'prediction': pred,
        'raw_probability': raw_prob,
        'calibrated_probability': calibrated_prob,
        'confidence': confidence,
        'uncertainty': uncertainty,
        'correct': pred == true_label,
        'filename': filename,
        'sagittal_importance': float(plane_importance[0]),
        'coronal_importance': float(plane_importance[1]),
        'axial_importance': float(plane_importance[2]),
        'top_region': all_regions[0] if all_regions else None
    }

def main():
    print("="*80)
    print("🔬 ENHANCED MULTI-PLANE XAI ANALYSIS WITH MEDICAL EXPLANATIONS")
    print("="*80)
    
    CONFIG = {
        'mrnet_path': '/Users/yuvrajpratapsingh/knee-mri-abnormality-detection/dataset/MRNet-v1.0',
        'model_path': '/Users/yuvrajpratapsingh/knee-mri-abnormality-detection/outputs/models/best_model_multiplane.pth',
        'output_dir': '/Users/yuvrajpratapsingh/knee-mri-abnormality-detection/outputs/xai_enhanced_medical',
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
    
    # Temperature scaling value from your evaluation
    temperature = 0.6617
    print(f"🌡️  Using temperature scaling: {temperature:.4f}")
    
    # Select diverse cases
    print("\n📊 Selecting diverse cases...")
    
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
            raw_prob = torch.sigmoid(output).item()
            calibrated_prob = torch.sigmoid(output / temperature).item()
            pred = 1 if calibrated_prob > 0.5 else 0
            
            all_predictions.append({
                'idx': idx,
                'case_id': val_dataset.labels_df.iloc[idx]['case'],
                'true_label': label,
                'prediction': pred,
                'raw_probability': raw_prob,
                'calibrated_probability': calibrated_prob,
                'correct': pred == label,
                'confidence': abs(calibrated_prob - 0.5) * 2
            })
    
    pred_df = pd.DataFrame(all_predictions)
    
    # Select interesting cases
    tp_high = pred_df[(pred_df['prediction'] == 1) & (pred_df['true_label'] == 1)].nlargest(3, 'confidence')
    tn_high = pred_df[(pred_df['prediction'] == 0) & (pred_df['true_label'] == 0)].nlargest(3, 'confidence')
    fp_all = pred_df[(pred_df['prediction'] == 1) & (pred_df['true_label'] == 0)]
    fn_all = pred_df[(pred_df['prediction'] == 0) & (pred_df['true_label'] == 1)]
    border_correct = pred_df[pred_df['correct'] == True].nsmallest(3, 'confidence')
    
    selected_indices = pd.concat([tp_high, tn_high, fp_all, fn_all, border_correct])['idx'].unique()
    
    print(f"\n🎯 Analyzing {len(selected_indices)} cases:")
    print(f"   - {len(tp_high)} high-confidence True Positives")
    print(f"   - {len(tn_high)} high-confidence True Negatives")
    print(f"   - {len(fp_all)} False Positives (ALL)")
    print(f"   - {len(fn_all)} False Negatives (ALL)")
    print(f"   - {len(border_correct)} Borderline correct cases")
    
    # Analyze cases
    results = []
    print("\n🔬 Generating enhanced visualizations...")
    
    for idx in tqdm(selected_indices):
        try:
            result = analyze_case_enhanced(model, val_dataset, idx, device, CONFIG['output_dir'])
            results.append(result)
        except Exception as e:
            print(f"\n⚠️  Error on case {idx}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{CONFIG['output_dir']}/enhanced_xai_summary.csv", index=False)
    
    # Print summary statistics
    print("\n" + "="*80)
    print("📊 ANALYSIS SUMMARY")
    print("="*80)
    
    print(f"\nCases Analyzed: {len(results)}")
    print(f"Correct Predictions: {results_df['correct'].sum()}/{len(results)} ({results_df['correct'].mean():.1%})")
    
    print(f"\nAverage Calibrated Probability:")
    print(f"  All cases: {results_df['calibrated_probability'].mean():.1%}")
    print(f"  Correct: {results_df[results_df['correct']==True]['calibrated_probability'].mean():.1%}")
    if len(results_df[results_df['correct']==False]) > 0:
        print(f"  Incorrect: {results_df[results_df['correct']==False]['calibrated_probability'].mean():.1%}")
    
    print(f"\nAverage Confidence:")
    print(f"  All cases: {results_df['confidence'].mean():.1%}")
    print(f"  Correct: {results_df[results_df['correct']==True]['confidence'].mean():.1%}")
    if len(results_df[results_df['correct']==False]) > 0:
        print(f"  Incorrect: {results_df[results_df['correct']==False]['confidence'].mean():.1%}")
    
    print(f"\nPlane Importance (Average):")
    print(f"  Sagittal: {results_df['sagittal_importance'].mean():.1%}")
    print(f"  Coronal:  {results_df['coronal_importance'].mean():.1%}")
    print(f"  Axial:    {results_df['axial_importance'].mean():.1%}")
    
    print("\n" + "="*80)
    print("✅ ENHANCED XAI ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\n📁 Results saved to: {CONFIG['output_dir']}/")
    print(f"   - {len(results)} detailed case visualizations")
    print(f"   - enhanced_xai_summary.csv with all metrics")
    print("="*80)
    
    print("\n💡 Key Features:")
    print("   ✓ Anatomical region identification")
    print("   ✓ Color-coded importance heatmaps")
    print("   ✓ Comprehensive interpretation legends")
    print("   ✓ Clinical decision explanations")
    print("   ✓ Temperature-scaled probabilities")
    print("   ✓ Confidence and uncertainty metrics")

if __name__ == '__main__':
    main()