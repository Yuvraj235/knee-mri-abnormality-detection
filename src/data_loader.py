import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image

class MRNetDataset(Dataset):
    """Enhanced MRNet Dataset with better augmentation"""
    
    def __init__(self, root_dir, plane='sagittal', task='abnormal', split='train', 
                 use_all_slices=False, transform=None):
        self.root_dir = root_dir
        self.plane = plane
        self.task = task
        self.split = split
        self.use_all_slices = use_all_slices
        
        # Load labels
        label_file = f'{split}-{task}.csv'
        label_path = os.path.join(root_dir, label_file)
        
        if not os.path.exists(label_path):
            label_file = f'{split}_{task}.csv'
            label_path = os.path.join(root_dir, label_file)
        
        self.labels_df = pd.read_csv(label_path, names=['case', 'label'], header=None)
        
        # Count class distribution
        pos_count = (self.labels_df['label'] == 1).sum()
        neg_count = (self.labels_df['label'] == 0).sum()
        print(f"📊 Loaded {len(self.labels_df)} cases from {split}/{plane}")
        print(f"   Task: {task}")
        print(f"   Positive cases: {pos_count}")
        print(f"   Negative cases: {neg_count}")
        
        # AGGRESSIVE augmentation for training
        if transform is None:
            if split == 'train':
                self.transform = transforms.Compose([
                    transforms.Resize((256, 256)),
                    transforms.RandomCrop(224),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(15),
                    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                    transforms.ColorJitter(brightness=0.3, contrast=0.3),
                    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                    transforms.RandomErasing(p=0.3, scale=(0.02, 0.15))
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
        else:
            self.transform = transform
    
    def __len__(self):
        return len(self.labels_df)
    
    def __getitem__(self, idx):
        case = self.labels_df.iloc[idx]['case']
        label = self.labels_df.iloc[idx]['label']
        
        # FIX: Use 4-digit zero-padding for filename
        volume_path = os.path.join(self.root_dir, self.split, self.plane, 
                                   f'{case:04d}.npy')
        
        volume = np.load(volume_path)
        
        # Select middle slice
        if self.use_all_slices:
            slice_idx = np.random.randint(0, volume.shape[0])
        else:
            slice_idx = volume.shape[0] // 2
        
        slice_img = volume[slice_idx]
        
        # Convert to PIL for transforms
        slice_img = (slice_img - slice_img.min()) / (slice_img.max() - slice_img.min() + 1e-8)
        slice_img = (slice_img * 255).astype(np.uint8)
        slice_img = Image.fromarray(slice_img).convert('RGB')
        
        # Apply transforms
        if self.transform:
            slice_img = self.transform(slice_img)
        
        return slice_img, torch.tensor(label, dtype=torch.float32)
