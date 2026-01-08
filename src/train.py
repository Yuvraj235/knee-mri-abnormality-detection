import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import MRNetDataset
from src.model import ResNetDeiTFusion

def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []
    
    pbar = tqdm(train_loader, desc=f'Training Epoch {epoch}')
    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        running_loss += loss.item()
        preds = torch.sigmoid(outputs).detach().cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        
        # Update progress bar
        avg_loss = running_loss / (batch_idx + 1)
        pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
    
    avg_loss = running_loss / len(train_loader)
    all_preds = np.array(all_preds).flatten()
    all_labels = np.array(all_labels).flatten()
    pred_binary = (all_preds > 0.5).astype(int)
    
    accuracy = accuracy_score(all_labels, pred_binary)
    auc = roc_auc_score(all_labels, all_preds)
    f1 = f1_score(all_labels, pred_binary, zero_division=0)
    
    return avg_loss, accuracy, auc, f1

def validate(model, val_loader, criterion, device, epoch):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f'Validation Epoch {epoch}'):
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            preds = torch.sigmoid(outputs).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = running_loss / len(val_loader)
    all_preds = np.array(all_preds).flatten()
    all_labels = np.array(all_labels).flatten()
    pred_binary = (all_preds > 0.5).astype(int)
    
    accuracy = accuracy_score(all_labels, pred_binary)
    auc = roc_auc_score(all_labels, all_preds)
    f1 = f1_score(all_labels, pred_binary, zero_division=0)
    
    num_pos_pred = pred_binary.sum()
    num_neg_pred = len(pred_binary) - num_pos_pred
    
    return avg_loss, accuracy, auc, f1, num_pos_pred, num_neg_pred

def plot_training_history(history, output_path):
    """Create comprehensive training plots"""
    fig = plt.figure(figsize=(20, 5))
    
    # Loss plot
    ax1 = plt.subplot(1, 4, 1)
    ax1.plot(history['train_loss'], label='Train Loss', linewidth=2, color='blue', marker='o')
    ax1.plot(history['val_loss'], label='Val Loss', linewidth=2, color='orange', marker='s')
    ax1.set_xlabel('Epoch', fontsize=12, weight='bold')
    ax1.set_ylabel('Loss', fontsize=12, weight='bold')
    ax1.set_title('Training & Validation Loss', fontsize=14, weight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # AUC plot
    ax2 = plt.subplot(1, 4, 2)
    ax2.plot(history['train_auc'], label='Train AUC', linewidth=2, color='green', marker='o')
    ax2.plot(history['val_auc'], label='Val AUC', linewidth=2, color='red', marker='s')
    ax2.axhline(y=0.5, color='gray', linestyle='--', label='Random', alpha=0.5)
    ax2.set_xlabel('Epoch', fontsize=12, weight='bold')
    ax2.set_ylabel('AUC', fontsize=12, weight='bold')
    ax2.set_title('AUC Score', fontsize=14, weight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0.4, 1.0])
    
    # Accuracy plot
    ax3 = plt.subplot(1, 4, 3)
    ax3.plot(history['val_acc'], label='Val Accuracy', linewidth=2, color='purple', marker='D')
    ax3.set_xlabel('Epoch', fontsize=12, weight='bold')
    ax3.set_ylabel('Accuracy', fontsize=12, weight='bold')
    ax3.set_title('Validation Accuracy', fontsize=14, weight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0.5, 1.0])
    
    # Learning Rate plot
    ax4 = plt.subplot(1, 4, 4)
    ax4.plot(history['lr'], label='Learning Rate', linewidth=2, color='brown', marker='x')
    ax4.set_xlabel('Epoch', fontsize=12, weight='bold')
    ax4.set_ylabel('Learning Rate', fontsize=12, weight='bold')
    ax4.set_title('Learning Rate Schedule', fontsize=14, weight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 Training plots saved to {output_path}")

def main():
    print("="*70)
    print("🦵 KNEE MRI ABNORMALITY DETECTION - TRAINING")
    print("="*70)
    
    CONFIG = {
        'mrnet_path': '/Users/yuvrajpratapsingh/knee-mri-abnormality-detection/dataset/MRNet-v1.0',
        'output_dir': '/Users/yuvrajpratapsingh/knee-mri-abnormality-detection/outputs',
        'plane': 'sagittal',
        'task': 'abnormal',
        'batch_size': 16,
        'num_epochs': 50,
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'fusion_type': 'concat',
        'num_workers': 2,
        'early_stopping_patience': 15,
    }
    
    # Create output directories
    os.makedirs(f"{CONFIG['output_dir']}/models", exist_ok=True)
    os.makedirs(f"{CONFIG['output_dir']}/plots", exist_ok=True)
    os.makedirs(f"{CONFIG['output_dir']}/logs", exist_ok=True)
    
    # Device setup
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\n🖥️  Using device: {device}")
    torch.set_default_dtype(torch.float32)
    
    # Load datasets
    print("\n📦 Loading datasets...")
    train_dataset = MRNetDataset(
        root_dir=CONFIG['mrnet_path'], plane=CONFIG['plane'], 
        task=CONFIG['task'], split='train', use_all_slices=False
    )
    val_dataset = MRNetDataset(
        root_dir=CONFIG['mrnet_path'], plane=CONFIG['plane'],
        task=CONFIG['task'], split='valid', use_all_slices=False
    )
    
    # Balanced sampler
    train_labels = train_dataset.labels_df['label'].values
    class_counts = np.bincount(train_labels)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[train_labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=CONFIG['batch_size'],
        sampler=sampler,
        num_workers=CONFIG['num_workers']
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=CONFIG['batch_size'],
        shuffle=False, 
        num_workers=CONFIG['num_workers']
    )
    
    print(f"✅ Training samples: {len(train_dataset)}")
    print(f"✅ Validation samples: {len(val_dataset)}")
    print(f"✅ Class distribution - Normal: {class_counts[0]}, Abnormal: {class_counts[1]}")
    print(f"✅ Using balanced sampling")
    
    # Create model
    print("\n🤖 Creating model...")
    model = ResNetDeiTFusion(
        num_classes=1, 
        fusion_type=CONFIG['fusion_type'],
        pretrained=True
    )
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✅ Total parameters: {total_params:,}")
    print(f"✅ Trainable parameters: {trainable_params:,}")
    
    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'], 
                          weight_decay=CONFIG['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CONFIG['num_epochs'], eta_min=1e-6
    )
    
    # Training loop
    print("\n" + "="*70)
    print(f"🚀 STARTING TRAINING - {CONFIG['num_epochs']} EPOCHS")
    print("="*70)
    
    best_val_auc = 0.0
    patience_counter = 0
    start_time = time.time()
    
    history = {
        'train_loss': [], 'val_loss': [], 
        'train_auc': [], 'val_auc': [], 
        'val_acc': [], 'lr': []
    }
    
    for epoch in range(CONFIG['num_epochs']):
        epoch_start = time.time()
        
        print(f"\n{'='*70}")
        print(f"📅 Epoch {epoch+1}/{CONFIG['num_epochs']}")
        print(f"{'='*70}")
        
        # Training
        train_loss, train_acc, train_auc, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch+1
        )
        
        # Validation
        val_loss, val_acc, val_auc, val_f1, num_pos, num_neg = validate(
            model, val_loader, criterion, device, epoch+1
        )
        
        # Update scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Store history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_auc'].append(train_auc)
        history['val_auc'].append(val_auc)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)
        
        # Print metrics
        epoch_time = time.time() - epoch_start
        print(f"\n📊 RESULTS:")
        print(f"   Train → Loss: {train_loss:.4f} | Acc: {train_acc:.3f} | AUC: {train_auc:.4f} | F1: {train_f1:.3f}")
        print(f"   Val   → Loss: {val_loss:.4f} | Acc: {val_acc:.3f} | AUC: {val_auc:.4f} | F1: {val_f1:.3f}")
        print(f"   Predictions → Abnormal: {num_pos}/120 | Normal: {num_neg}/120")
        print(f"   LR: {current_lr:.2e} | Time: {epoch_time:.1f}s")
        
        # Check for improvement
        if val_auc > best_val_auc:
            improvement = val_auc - best_val_auc
            best_val_auc = val_auc
            patience_counter = 0
            
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_auc': val_auc,
                'val_acc': val_acc,
                'val_f1': val_f1,
                'train_auc': train_auc,
                'config': CONFIG,
                'history': history
            }, f"{CONFIG['output_dir']}/models/best_model.pth")
            
            print(f"   ✅ NEW BEST MODEL! AUC: {val_auc:.4f} (+{improvement:.4f})")
        else:
            patience_counter += 1
            print(f"   ⏸️  No improvement ({patience_counter}/{CONFIG['early_stopping_patience']})")
            
            if patience_counter >= CONFIG['early_stopping_patience']:
                print(f"\n⏹️  EARLY STOPPING triggered after {epoch+1} epochs")
                print(f"   Best validation AUC: {best_val_auc:.4f}")
                break
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auc': val_auc,
            }, f"{CONFIG['output_dir']}/models/checkpoint_epoch_{epoch+1}.pth")
            print(f"   �� Checkpoint saved: epoch_{epoch+1}.pth")
    
    # Training complete
    total_time = time.time() - start_time
    print("\n" + "="*70)
    print("�� TRAINING COMPLETE!")
    print("="*70)
    print(f"✅ Best Validation AUC: {best_val_auc:.4f}")
    print(f"✅ Total Training Time: {total_time/60:.1f} minutes")
    print(f"✅ Best model saved to: {CONFIG['output_dir']}/models/best_model.pth")
    print("Training complete successfully and the best model is saved in the models folder with the name best_model.pth and the training history is saved in the plots folder with the name training_history ")
    # Plot training history
    plot_path = f"{CONFIG['output_dir']}/plots/training_history.png"
    plot_training_history(history, plot_path)
    
    # Save training log
    log_path = f"{CONFIG['output_dir']}/logs/training_log.txt"
    with open(log_path, 'w') as f:
        f.write("TRAINING LOG\n")
        f.write("="*70 + "\n\n")
        f.write(f"Configuration:\n")
        for key, value in CONFIG.items():
            f.write(f"  {key}: {value}\n")
        f.write(f"\nBest Validation AUC: {best_val_auc:.4f}\n")
        f.write(f"Total Training Time: {total_time/60:.1f} minutes\n")
        f.write(f"\nEpoch-by-Epoch Results:\n")
        f.write("-"*70 + "\n")
        for i in range(len(history['val_auc'])):
            f.write(f"Epoch {i+1:2d} | Train AUC: {history['train_auc'][i]:.4f} | ")
            f.write(f"Val AUC: {history['val_auc'][i]:.4f} | Val Acc: {history['val_acc'][i]:.4f}\n")
    
    print(f"📝 Training log saved to: {log_path}")
    print("\n" + "="*70)
    print("🎯 NEXT STEPS:")
    print("="*70)
    print("1. Run evaluation: python src/evaluate.py")
    print("2. View plots: open outputs/plots/training_history.png")
    print("3. Check logs: cat outputs/logs/training_log.txt")
    print("="*70)

if __name__ == '__main__':
    main()
