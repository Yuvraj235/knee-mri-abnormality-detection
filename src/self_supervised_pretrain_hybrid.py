import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.fastmri_loader import FastMRIKneeDataset
from src.hybrid_cnn_transformer_model import MultiPlaneHybridFusion

class ContrastiveAugmentation:
    def __init__(self, size=(224, 224)):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def __call__(self, x):
        return self.transform(x), self.transform(x)


class ContrastiveFastMRIDataset(FastMRIKneeDataset):
    def __init__(self, root_dir, split='train'):
        super().__init__(root_dir, split, transform=None)
        self.augmentation = ContrastiveAugmentation()
    
    def __getitem__(self, idx):
        h5_path = self.file_list[idx]
        
        try:
            import h5py
            with h5py.File(h5_path, 'r') as f:
                volume = f['reconstruction_rss'][:]
        except:
            volume = np.zeros((30, 320, 320))
        
        n_slices = volume.shape[0]
        sag_slice = volume[n_slices // 2]
        cor_slice = volume[n_slices // 3]
        axi_slice = volume[(2 * n_slices) // 3]
        
        sag_img = self._to_pil_image(sag_slice)
        cor_img = self._to_pil_image(cor_slice)
        axi_img = self._to_pil_image(axi_slice)
        
        sag_v1, sag_v2 = self.augmentation(sag_img)
        cor_v1, cor_v2 = self.augmentation(cor_img)
        axi_v1, axi_v2 = self.augmentation(axi_img)
        
        return {
            'view1': {'sagittal': sag_v1, 'coronal': cor_v1, 'axial': axi_v1},
            'view2': {'sagittal': sag_v2, 'coronal': cor_v2, 'axial': axi_v2}
        }


class ContrastiveModel(nn.Module):
    def __init__(self, base_model, projection_dim=128):
        super().__init__()
        self.encoder = base_model
        feature_dim = 128
        
        self.projection_head = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, projection_dim)
        )
    
    def forward(self, sagittal, coronal, axial):
        features = self.encoder.get_features(sagittal, coronal, axial)
        projections = self.projection_head(features)
        projections = F.normalize(projections, dim=1)
        return projections


class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, z_i, z_j):
        batch_size = z_i.shape[0]
        z = torch.cat([z_i, z_j], dim=0)
        sim_matrix = torch.mm(z, z.T) / self.temperature
        
        labels = torch.cat([
            torch.arange(batch_size) + batch_size,
            torch.arange(batch_size)
        ]).to(z.device)
        
        mask = torch.eye(2 * batch_size, dtype=torch.bool).to(z.device)
        sim_matrix = sim_matrix.masked_fill(mask, -9e15)
        
        loss = F.cross_entropy(sim_matrix, labels)
        return loss


def train_epoch(model, loader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0
    
    pbar = tqdm(loader, desc=f'Epoch {epoch}')
    for batch in pbar:
        view1 = batch['view1']
        view2 = batch['view2']
        
        sag1 = view1['sagittal'].to(device)
        cor1 = view1['coronal'].to(device)
        axi1 = view1['axial'].to(device)
        
        sag2 = view2['sagittal'].to(device)
        cor2 = view2['coronal'].to(device)
        axi2 = view2['axial'].to(device)
        
        z1 = model(sag1, cor1, axi1)
        z2 = model(sag2, cor2, axi2)
        
        loss = criterion(z1, z2)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(loader)


def main():
    print("="*70)
    print("🧠 PHASE 1: SELF-SUPERVISED PRE-TRAINING")
    print("   Model: Hybrid CNN-Transformer (ResNet50 + DeiT-Tiny)")
    print("   Dataset: fastMRI (973 unlabeled volumes)")
    print("="*70)
    
    config = {
        'data_dir': '/Users/yuvrajpratapsingh/knee-mri-abnormality-detection/dataset/fastmri_knee',
        'save_dir': '/Users/yuvrajpratapsingh/knee-mri-abnormality-detection/outputs/self_supervised_hybrid',
        'batch_size': 8,
        'learning_rate': 0.0005,
        'epochs': 10,
    }
    
    os.makedirs(config['save_dir'], exist_ok=True)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\n🖥️  Device: {device}")
    
    print("\n📦 Loading dataset...")
    train_dataset = ContrastiveFastMRIDataset(config['data_dir'], split='train')
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], 
                              shuffle=True, num_workers=2)
    
    print("\n🤖 Creating model...")
    base_model = MultiPlaneHybridFusion(num_classes=1, dropout_rate=0.3)
    model = ContrastiveModel(base_model, projection_dim=128)
    model = model.to(device)
    
    criterion = NTXentLoss(temperature=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    print(f"\n🚀 Starting pre-training for {config['epochs']} epochs...")
    print("⏰ This will take 6-8 hours. Let it run!\n")
    
    best_loss = float('inf')
    
    for epoch in range(1, config['epochs'] + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, epoch)
        
        print(f"\nEpoch {epoch}/{config['epochs']} - Loss: {train_loss:.4f}")
        
        if train_loss < best_loss:
            best_loss = train_loss
            torch.save({
                'epoch': epoch,
                'encoder_state_dict': model.encoder.state_dict(),
                'loss': train_loss,
            }, f"{config['save_dir']}/pretrained_hybrid_best.pth")
            print(f"✅ Saved best model!")
    
    print("\n" + "="*70)
    print("✅ PHASE 1 COMPLETE!")
    print(f"📁 Saved to: {config['save_dir']}/pretrained_hybrid_best.pth")
    print("="*70)


if __name__ == '__main__':
    main()
