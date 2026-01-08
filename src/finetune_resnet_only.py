import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.multiplane_loader import MultiPlaneMRNetDataset
from src.multiplane_model import MultiPlaneFusion

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in tqdm(loader, desc='Training'):
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
    
    return total_loss / len(loader), correct / total


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(loader, desc='Validating'):
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
    
    return total_loss / len(loader), correct / total


def main():
    print("="*70)
    print("🎯 TRAINING: ResNet50 Only (Simpler Model)")
    print("   No Transformer - Just CNN")
    print("="*70)
    
    config = {
        'mrnet_dir': '/Users/yuvrajpratapsingh/knee-mri-abnormality-detection/dataset/MRNet-v1.0',
        'save_dir': '/Users/yuvrajpratapsingh/knee-mri-abnormality-detection/outputs/resnet_only',
        'batch_size': 8,
        'learning_rate': 0.0001,
        'epochs': 25,
    }
    
    os.makedirs(config['save_dir'], exist_ok=True)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\n🖥️  Device: {device}")
    
    # Load MRNet
    print("\n📦 Loading MRNet...")
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
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # Create simpler model (ResNet50 only)
    print("\n🤖 Creating ResNet50 model...")
    model = MultiPlaneFusion(num_classes=1, dropout_rate=0.4)
    model = model.to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    print(f"\n🚀 Training for {config['epochs']} epochs...\n")
    
    best_val_acc = 0
    
    for epoch in range(1, config['epochs'] + 1):
        print(f"\nEpoch {epoch}/{config['epochs']}")
        
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        print(f"Train: Loss={train_loss:.4f}, Acc={train_acc:.4f}")
        print(f"Val:   Loss={val_loss:.4f}, Acc={val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
            }, f"{config['save_dir']}/best_model.pth")
            print(f"✅ Best: {val_acc:.4f}")
    
    print("\n" + "="*70)
    print(f"✅ COMPLETE! Best Val Acc: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    print("="*70)


if __name__ == '__main__':
    main()
