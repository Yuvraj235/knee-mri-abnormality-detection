import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.multiplane_loader import MultiPlaneMRNetDataset
from src.multiplane_model import MultiPlaneFusion

class TemperatureScaling(nn.Module):
    """
    Temperature scaling for model calibration
    """
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
    
    def forward(self, logits):
        return logits / self.temperature
    
    def fit(self, logits, labels, lr=0.01, max_iter=100):
        """
        Tune temperature on validation set
        """
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
        
        def eval():
            optimizer.zero_grad()
            loss = criterion(self.forward(logits), labels)
            loss.backward()
            return loss
        
        optimizer.step(eval)
        return self.temperature.item()

def analyze_calibration_problem(model, dataset, device):
    """
    Diagnose why your model has low probabilities
    """
    
    print("\n" + "="*70)
    print("🔍 DIAGNOSING CALIBRATION ISSUE")
    print("="*70)
    
    model.eval()
    
    all_logits = []
    all_probs = []
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for idx in range(len(dataset)):
            batch = dataset[idx]
            sag = batch['sagittal'].unsqueeze(0).to(device)
            cor = batch['coronal'].unsqueeze(0).to(device)
            axi = batch['axial'].unsqueeze(0).to(device)
            label = batch['label']
            
            output = model(sag, cor, axi)
            prob = torch.sigmoid(output)
            pred = (prob > 0.5).float()
            
            all_logits.append(output.cpu())
            all_probs.append(prob.cpu())
            all_labels.append(label)
            all_preds.append(pred.cpu())
    
    all_logits = torch.cat(all_logits)
    all_probs = torch.cat(all_probs).squeeze().numpy()
    all_labels = torch.stack(all_labels).squeeze().numpy()
    all_preds = torch.cat(all_preds).squeeze().numpy()
    
    # Calculate metrics
    accuracy = (all_preds == all_labels).mean()
    
    # Calibration metrics
    prob_true, prob_pred = calibration_curve(all_labels, all_probs, n_bins=10, strategy='uniform')
    brier = brier_score_loss(all_labels, all_probs)
    
    # Expected Calibration Error (ECE)
    ece = np.mean(np.abs(prob_true - prob_pred))
    
    print(f"\n📊 Current Model Performance:")
    print(f"   Accuracy: {accuracy:.2%}")
    print(f"   Brier Score: {brier:.4f} (lower is better)")
    print(f"   ECE: {ece:.4f} (lower is better)")
    
    # Probability distribution analysis
    print(f"\n📈 Probability Distribution:")
    print(f"   Mean probability: {all_probs.mean():.3f}")
    print(f"   Std probability: {all_probs.std():.3f}")
    print(f"   Median: {np.median(all_probs):.3f}")
    print(f"   Min: {all_probs.min():.3f}")
    print(f"   Max: {all_probs.max():.3f}")
    
    # Check if probabilities cluster around 0.5
    near_50 = np.sum((all_probs > 0.4) & (all_probs < 0.6)) / len(all_probs)
    print(f"   % near 0.5 (0.4-0.6): {near_50:.1%}")
    
    if near_50 > 0.5:
        print("\n⚠️  DIAGNOSIS: Your model is UNDER-CONFIDENT")
        print("   Most predictions cluster around 50%")
        print("   This is a CALIBRATION problem, not accuracy problem")
    
    # Visualize calibration
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Calibration curve
    axes[0].plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
    axes[0].plot(prob_pred, prob_true, 's-', label='Model')
    axes[0].set_xlabel('Mean Predicted Probability', fontsize=12)
    axes[0].set_ylabel('Fraction of Positives', fontsize=12)
    axes[0].set_title(f'Calibration Curve\nECE: {ece:.4f}', fontsize=14, weight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Probability histogram
    axes[1].hist(all_probs, bins=20, alpha=0.7, edgecolor='black')
    axes[1].axvline(0.5, color='red', linestyle='--', linewidth=2, label='Decision threshold')
    axes[1].set_xlabel('Predicted Probability', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('Probability Distribution', fontsize=14, weight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    # Confidence by correctness
    correct_probs = all_probs[all_preds == all_labels]
    incorrect_probs = all_probs[all_preds != all_labels]
    
    axes[2].hist([correct_probs, incorrect_probs], 
                bins=20, 
                alpha=0.7, 
                label=['Correct', 'Incorrect'],
                edgecolor='black')
    axes[2].axvline(0.5, color='red', linestyle='--', linewidth=2)
    axes[2].set_xlabel('Predicted Probability', fontsize=12)
    axes[2].set_ylabel('Frequency', fontsize=12)
    axes[2].set_title('Confidence by Correctness', fontsize=14, weight='bold')
    axes[2].legend()
    axes[2].grid(alpha=0.3)
    
    plt.tight_layout()
    
    return {
        'logits': all_logits,
        'probs': all_probs,
        'labels': torch.tensor(all_labels),
        'accuracy': accuracy,
        'ece': ece,
        'brier': brier,
        'fig': fig
    }

def apply_temperature_scaling(logits, labels):
    """
    Find optimal temperature to calibrate probabilities
    """
    
    print("\n🌡️  Applying Temperature Scaling...")
    
    temp_model = TemperatureScaling()
    optimal_temp = temp_model.fit(logits, labels.float().unsqueeze(1))
    
    print(f"   Optimal temperature: {optimal_temp:.4f}")
    
    # Apply temperature
    calibrated_logits = logits / optimal_temp
    calibrated_probs = torch.sigmoid(calibrated_logits).numpy()
    
    # Recalculate metrics
    calibrated_preds = (calibrated_probs > 0.5).astype(float)
    calibrated_accuracy = (calibrated_preds.flatten() == labels.numpy()).mean()
    
    prob_true, prob_pred = calibration_curve(labels.numpy(), calibrated_probs.flatten(), n_bins=10)
    calibrated_ece = np.mean(np.abs(prob_true - prob_pred))
    calibrated_brier = brier_score_loss(labels.numpy(), calibrated_probs.flatten())
    
    print(f"\n📊 After Temperature Scaling:")
    print(f"   Accuracy: {calibrated_accuracy:.2%}")
    print(f"   ECE: {calibrated_ece:.4f}")
    print(f"   Brier: {calibrated_brier:.4f}")
    print(f"   Mean probability: {calibrated_probs.mean():.3f}")
    print(f"   Std probability: {calibrated_probs.std():.3f}")
    
    return optimal_temp, calibrated_probs

def save_calibrated_model(model, temperature, save_path):
    """
    Save model with calibration temperature
    """
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'temperature': temperature,
        'calibrated': True
    }
    
    torch.save(checkpoint, save_path)
    print(f"\n💾 Saved calibrated model to: {save_path}")

def main():
    print("="*70)
    print("🔧 MODEL CALIBRATION FIX")
    print("="*70)
    
    CONFIG = {
        'mrnet_path': '/Users/yuvrajpratapsingh/knee-mri-abnormality-detection/dataset/MRNet-v1.0',
        'model_path': '/Users/yuvrajpratapsingh/knee-mri-abnormality-detection/outputs/models/best_model_multiplane.pth',
        'output_dir': '/Users/yuvrajpratapsingh/knee-mri-abnormality-detection/outputs/calibration',
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
    print("\n🤖 Loading model...")
    model = MultiPlaneFusion(num_classes=1, dropout_rate=0.4)
    checkpoint = torch.load(CONFIG['model_path'], map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    # Analyze calibration issue
    calibration_data = analyze_calibration_problem(model, val_dataset, device)
    
    # Save diagnosis plot
    calibration_data['fig'].savefig(
        os.path.join(CONFIG['output_dir'], 'calibration_diagnosis.png'),
        dpi=150,
        bbox_inches='tight'
    )
    plt.close()
    
    # Apply temperature scaling
    optimal_temp, calibrated_probs = apply_temperature_scaling(
        calibration_data['logits'],
        calibration_data['labels']
    )
    
    # Visualize improvement
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Before vs After histograms
    axes[0].hist(calibration_data['probs'], bins=20, alpha=0.6, label='Before', edgecolor='black')
    axes[0].hist(calibrated_probs, bins=20, alpha=0.6, label='After', edgecolor='black')
    axes[0].axvline(0.5, color='red', linestyle='--', linewidth=2)
    axes[0].set_xlabel('Predicted Probability', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Probability Distribution: Before vs After', fontsize=14, weight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Calibration curves comparison
    prob_true_before, prob_pred_before = calibration_curve(
        calibration_data['labels'].numpy(),
        calibration_data['probs'],
        n_bins=10
    )
    prob_true_after, prob_pred_after = calibration_curve(
        calibration_data['labels'].numpy(),
        calibrated_probs.flatten(),
        n_bins=10
    )
    
    axes[1].plot([0, 1], [0, 1], 'k--', label='Perfect', linewidth=2)
    axes[1].plot(prob_pred_before, prob_true_before, 's-', label='Before', linewidth=2)
    axes[1].plot(prob_pred_after, prob_true_after, 'o-', label='After', linewidth=2)
    axes[1].set_xlabel('Mean Predicted Probability', fontsize=12)
    axes[1].set_ylabel('Fraction of Positives', fontsize=12)
    axes[1].set_title('Calibration Improvement', fontsize=14, weight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(
        os.path.join(CONFIG['output_dir'], 'calibration_improvement.png'),
        dpi=150,
        bbox_inches='tight'
    )
    plt.close()
    
    # Save calibrated model
    calibrated_model_path = os.path.join(CONFIG['output_dir'], 'calibrated_model.pth')
    
    checkpoint_with_temp = {
        'model_state_dict': model.state_dict(),
        'temperature': optimal_temp,
        'epoch': checkpoint.get('epoch', 0),
        'best_val_auc': checkpoint.get('best_val_auc', 0),
        'calibrated': True
    }
    
    torch.save(checkpoint_with_temp, calibrated_model_path)
    
    print("\n" + "="*70)
    print("✅ CALIBRATION COMPLETE!")
    print("="*70)
    print(f"\n📊 Improvement Summary:")
    print(f"   ECE: {calibration_data['ece']:.4f} → {np.mean(np.abs(prob_true_after - prob_pred_after)):.4f}")
    print(f"   Brier: {calibration_data['brier']:.4f} → {brier_score_loss(calibration_data['labels'].numpy(), calibrated_probs.flatten()):.4f}")
    
    print(f"\n💾 Saved:")
    print(f"   - Calibrated model: {calibrated_model_path}")
    print(f"   - Diagnosis plot: calibration_diagnosis.png")
    print(f"   - Improvement plot: calibration_improvement.png")
    
    print(f"\n🔧 Usage:")
    print(f"   To use calibrated predictions:")
    print(f"   output = model(inputs)")
    print(f"   calibrated_output = output / {optimal_temp:.4f}")
    print(f"   calibrated_prob = torch.sigmoid(calibrated_output)")
    
    print("\n" + "="*70)
    print("💡 Why This Fixes Low Probabilities:")
    print("="*70)
    print("Your model was trained with:")
    print("  • Cross-entropy loss")
    print("  • Dropout regularization")
    print("  • Early stopping")
    print("\nThis made it UNDER-CONFIDENT (probabilities near 0.5)")
    print("Temperature scaling STRETCHES probabilities to better match reality")
    print("Accuracy stays the same, but confidence improves!")

if __name__ == '__main__':
    main()