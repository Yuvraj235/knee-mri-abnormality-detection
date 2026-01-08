import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.multiplane_loader import MultiPlaneMRNetDataset
from src.hybrid_cnn_transformer_model import MultiPlaneHybridFusion

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc='Training')
    for batch in pbar:
        sag = batch['sagittal'].to(device)
        cor = batch['coronal'].to(device)
        axi = batch['axial'].to(device)
        labels = batch['label'].float().to(device)
        
        optimizer.zero_grad()
        outputs = model(sag, cor, axi).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        preds = (torch.sigmoid(outputs) > 0.5).long()
        correct += (preds == labels.long()).sum().item()
        total += labels.size(0)
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{correct/total:.4f}'})
    
    return total_loss / len(loader), correct / total


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(loader, desc='Validating')
        for batch in pbar:
            sag = batch['sagittal'].to(device)
            cor = batch['coronal'].to(device)
            axi = batch['axial'].to(device)
            labels = batch['label'].float().to(device)
            
            outputs = model(sag, cor, axi).squeeze()
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).long()
            correct += (preds == labels.long()).sum().item()
            total += labels.size(0)
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{correct/total:.4f}'})
    
    return total_loss / len(loader), correct / total


def main():
    print("="*70)
    print("🎯 PHASE 2: SUPERVISED FINE-TUNING")
    print("   Model: Pre-trained Hybrid CNN-Transformer")
    print("   Dataset: MRNet (1,130 labeled cases)")
    print("   Task: Abnormality Detection")
    print("="*70)
    
    config = {
        'mrnet_dir': '/Users/yuvrajpratapsingh/knee-mri-abnormality-detection/dataset/MRNet-v1.0',
        'pretrain_path': '/Users/yuvrajpratapsingh/knee-mri-abnormality-detection/outputs/self_supervised_hybrid/pretrained_hybrid_best.pth',
        'save_dir': '/Users/yuvrajpratapsingh/knee-mri-abnormality-detection/outputs/finetuned_hybrid',
        'batch_size': 8,
        'learning_rate': 0.0001,
        'epochs': 20,
    }
    
    os.makedirs(config['save_dir'], exist_ok=True)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\n🖥️  Device: {device}")
    
    # Load MRNet
    print("\n📦 Loading MRNet dataset...")
    train_dataset = MultiPlaneMRNetDataset(
        root_dir=config['mrnet_dir'],
        task='abnormal',
        split='train',
        use_all_slices=False
    )
    
    val_dataset = MultiPlaneMRNetDataset(
        root_dir=config['mrnet_dir'],
        task='abnormal',
        split='valid',
        use_all_slices=False
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=2)
    
    # Load model with pre-trained weights
    print("\n🤖 Loading pre-trained model...")
    model = MultiPlaneHybridFusion(num_classes=1, dropout_rate=0.4)
    
    if os.path.exists(config['pretrain_path']):
        checkpoint = torch.load(config['pretrain_path'], map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['encoder_state_dict'], strict=False)
        print(f"✅ Loaded pre-trained weights from epoch {checkpoint['epoch']}")
        print(f"   Pre-training loss: {checkpoint['loss']:.4f}")
    else:
        print("⚠️  No pre-trained weights found, training from scratch")
    
    model = model.to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    print(f"\n🚀 Fine-tuning for {config['epochs']} epochs...")
    print("⏰ Estimated time: 2-3 hours\n")
    
    best_val_acc = 0
    
    for epoch in range(1, config['epochs'] + 1):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch}/{config['epochs']}")
        print(f"{'='*70}")
        
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        print(f"\nResults:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}   | Val Acc:   {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, f"{config['save_dir']}/finetuned_hybrid_best.pth")
            print(f"  ✅ New best model! (Val Acc: {val_acc:.4f})")
    
    print("\n" + "="*70)
    print("✅ PHASE 2 COMPLETE!")
    print(f"🏆 Best Validation Accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    print(f"📁 Saved to: {config['save_dir']}/finetuned_hybrid_best.pth")
    print("="*70)


if __name__ == '__main__':
    main()
