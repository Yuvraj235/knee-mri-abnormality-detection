import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import MRNetDataset
from src.model import ImprovedResNetDeiTFusion

# Set default dtype
torch.set_default_dtype(torch.float32)

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        BCE_loss = nn.functional.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return F_loss.mean()

def calculate_metrics(predictions, labels):
    """Calculate sensitivity and specificity"""
    preds_binary = (predictions > 0.5).astype(int)
    labels_binary = labels.astype(int)
    
    tp = ((preds_binary == 1) & (labels_binary == 1)).sum()
    tn = ((preds_binary == 0) & (labels_binary == 0)).sum()
    fp = ((preds_binary == 1) & (labels_binary == 0)).sum()
    fn = ((preds_binary == 0) & (labels_binary == 1)).sum()
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return sensitivity, specificity

def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1} [Train]')
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.unsqueeze(1).to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        running_loss += loss.item()
        
        # Collect predictions
        probs = torch.sigmoid(outputs).detach().cpu().numpy()
        all_preds.extend(probs.flatten())
        all_labels.extend(labels.cpu().numpy().flatten())
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    epoch_loss = running_loss / len(train_loader)
    epoch_auc = roc_auc_score(all_labels, all_preds)
    sensitivity, specificity = calculate_metrics(np.array(all_preds), np.array(all_labels))
    
    return epoch_loss, epoch_auc, sensitivity, specificity

def validate_epoch(model, val_loader, criterion, device, epoch):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f'Epoch {epoch+1} [Valid]')
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.unsqueeze(1).to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            
            probs = torch.sigmoid(outputs).cpu().numpy()
            all_preds.extend(probs.flatten())
            all_labels.extend(labels.cpu().numpy().flatten())
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    epoch_loss = running_loss / len(val_loader)
    epoch_auc = roc_auc_score(all_labels, all_preds)
    epoch_acc = accuracy_score(all_labels, (np.array(all_preds) > 0.5).astype(int))
    sensitivity, specificity = calculate_metrics(np.array(all_preds), np.array(all_labels))
    
    return epoch_loss, epoch_auc, epoch_acc, sensitivity, specificity

def main():
    print("="*70)
    print("🚀 IMPROVED TRAINING - KNEE MRI ABNORMALITY DETECTION")
    print("="*70)
    
    CONFIG = {
        'mrnet_path': '/Users/yuvrajpratapsingh/knee-mri-abnormality-detection/dataset/MRNet-v1.0',
        'output_dir': '/Users/yuvrajpratapsingh/knee-mri-abnormality-detection/outputs',
        'plane': 'sagittal',
        'task': 'abnormal',
        'batch_size': 12,
        'num_epochs': 30,
        'learning_rate': 3e-5,
        'weight_decay': 1e-3,
        'dropout_rate': 0.5,
        'early_stopping_patience': 10,
        'min_specificity': 0.6,
    }
    
    os.makedirs(f"{CONFIG['output_dir']}/models", exist_ok=True)
    os.makedirs(f"{CONFIG['output_dir']}/plots", exist_ok=True)
    os.makedirs(f"{CONFIG['output_dir']}/logs", exist_ok=True)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\n🖥️  Using device: {device}")
    
    # Load datasets
    print("\n📦 Loading datasets...")
    train_dataset = MRNetDataset(
        root_dir=CONFIG['mrnet_path'],
        plane=CONFIG['plane'],
        task=CONFIG['task'],
        split='train',
        use_all_slices=True
    )
    
    val_dataset = MRNetDataset(
        root_dir=CONFIG['mrnet_path'],
        plane=CONFIG['plane'],
        task=CONFIG['task'],
        split='valid',
        use_all_slices=False
    )
    
    # Balanced sampling
    train_labels = train_dataset.labels_df['label'].values
    class_counts = np.bincount(train_labels.astype(int))
    class_weights = 1. / class_counts
    sample_weights = class_weights[train_labels.astype(int)]
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        sampler=sampler,
        num_workers=0,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    print(f"✅ Training batches: {len(train_loader)}")
    print(f"✅ Validation batches: {len(val_loader)}")
    
    # Model
    print("\n🤖 Creating model...")
    model = ImprovedResNetDeiTFusion(
        num_classes=1,
        fusion_type='concat',
        pretrained=True,
        dropout_rate=CONFIG['dropout_rate']
    )
    model = model.to(device)
    
    # Class weight calculation (FIX: Use float32)
    pos_weight_value = float(class_counts[0]) / float(class_counts[1])
    print(f"\n⚖️  Class weights: Normal=1.0, Abnormal={pos_weight_value:.2f}")
    
    # Use Focal Loss
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay']
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=CONFIG['num_epochs'],
        eta_min=1e-6
    )
    
    # Training history
    history = {
        'train_loss': [], 'val_loss': [],
        'train_auc': [], 'val_auc': [],
        'val_acc': [],
        'train_sensitivity': [], 'train_specificity': [],
        'val_sensitivity': [], 'val_specificity': [],
        'learning_rate': []
    }
    
    best_score = 0
    best_epoch = 0
    patience_counter = 0
    
    print("\n" + "="*70)
    print("🔥 Starting Training")
    print("="*70)
    
    start_time = time.time()
    
    for epoch in range(CONFIG['num_epochs']):
        # Train
        train_loss, train_auc, train_sens, train_spec = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validate
        val_loss, val_auc, val_acc, val_sens, val_spec = validate_epoch(
            model, val_loader, criterion, device, epoch
        )
        
        # Update scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_auc'].append(train_auc)
        history['val_auc'].append(val_auc)
        history['val_acc'].append(val_acc)
        history['train_sensitivity'].append(train_sens)
        history['train_specificity'].append(train_spec)
        history['val_sensitivity'].append(val_sens)
        history['val_specificity'].append(val_spec)
        history['learning_rate'].append(current_lr)
        
        # Print epoch summary
        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{CONFIG['num_epochs']}")
        print(f"{'='*70}")
        print(f"Train | Loss: {train_loss:.4f} | AUC: {train_auc:.4f} | "
              f"Sens: {train_sens:.3f} | Spec: {train_spec:.3f}")
        print(f"Valid | Loss: {val_loss:.4f} | AUC: {val_auc:.4f} | Acc: {val_acc:.4f} | "
              f"Sens: {val_sens:.3f} | Spec: {val_spec:.3f}")
        print(f"LR: {current_lr:.2e}")
        
        # Combined score: 50% AUC + 25% Sensitivity + 25% Specificity
        combined_score = 0.5 * val_auc + 0.25 * val_sens + 0.25 * val_spec
        
        # Save best model
        if combined_score > best_score and val_spec >= CONFIG['min_specificity']:
            best_score = combined_score
            best_epoch = epoch
            patience_counter = 0
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auc': val_auc,
                'val_acc': val_acc,
                'val_sensitivity': val_sens,
                'val_specificity': val_spec,
                'combined_score': combined_score,
            }
            
            torch.save(checkpoint, f"{CONFIG['output_dir']}/models/best_model_improved.pth")
            print(f"✅ New best model saved! (Score: {combined_score:.4f})")
        else:
            patience_counter += 1
            print(f"⏳ No improvement ({patience_counter}/{CONFIG['early_stopping_patience']})")
        
        # Early stopping
        if patience_counter >= CONFIG['early_stopping_patience']:
            print(f"\n⏹️  Early stopping triggered after {epoch+1} epochs")
            break
    
    elapsed_time = time.time() - start_time
    print(f"\n✅ Training completed in {elapsed_time/60:.2f} minutes")
    print(f"🏆 Best model from epoch {best_epoch+1} with score {best_score:.4f}")
    
    # Plot history
    print("\n📊 Creating training plots...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    epochs_range = range(1, len(history['train_loss']) + 1)
    
    # Loss
    axes[0, 0].plot(epochs_range, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0, 0].plot(epochs_range, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0, 0].set_title('Loss', fontsize=14, weight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # AUC
    axes[0, 1].plot(epochs_range, history['train_auc'], 'b-', label='Train AUC', linewidth=2)
    axes[0, 1].plot(epochs_range, history['val_auc'], 'r-', label='Val AUC', linewidth=2)
    axes[0, 1].axhline(y=0.5, color='gray', linestyle='--', label='Random')
    axes[0, 1].set_title('AUC Score', fontsize=14, weight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('AUC')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # Validation Accuracy
    axes[0, 2].plot(epochs_range, history['val_acc'], 'm-', linewidth=2)
    axes[0, 2].set_title('Validation Accuracy', fontsize=14, weight='bold')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Accuracy')
    axes[0, 2].grid(alpha=0.3)
    
    # Sensitivity
    axes[1, 0].plot(epochs_range, history['train_sensitivity'], 'b-', label='Train', linewidth=2)
    axes[1, 0].plot(epochs_range, history['val_sensitivity'], 'r-', label='Val', linewidth=2)
    axes[1, 0].set_title('Sensitivity (Recall)', fontsize=14, weight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Sensitivity')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # Specificity
    axes[1, 1].plot(epochs_range, history['train_specificity'], 'b-', label='Train', linewidth=2)
    axes[1, 1].plot(epochs_range, history['val_specificity'], 'r-', label='Val', linewidth=2)
    axes[1, 1].axhline(y=0.6, color='orange', linestyle='--', label='Min Target', alpha=0.7)
    axes[1, 1].set_title('Specificity', fontsize=14, weight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Specificity')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    
    # Learning Rate
    axes[1, 2].plot(epochs_range, history['learning_rate'], 'g-', linewidth=2)
    axes[1, 2].set_title('Learning Rate Schedule', fontsize=14, weight='bold')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Learning Rate')
    axes[1, 2].set_yscale('log')
    axes[1, 2].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{CONFIG['output_dir']}/plots/training_history_improved.png", dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Training plots saved")
    print("\n" + "="*70)
    print("🎉 TRAINING COMPLETE!")
    print("="*70)

if __name__ == '__main__':
    main()
