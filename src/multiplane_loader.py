import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image

class MultiPlaneMRNetDataset(Dataset):
    """
    INNOVATION: Load all 3 MRI planes simultaneously
    This mimics how radiologists actually read MRIs!
    """
    
    def __init__(self, root_dir, task='abnormal', split='train', use_all_slices=False):
        self.root_dir = root_dir
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
        
        # Count distribution
        pos_count = (self.labels_df['label'] == 1).sum()
        neg_count = (self.labels_df['label'] == 0).sum()
        print(f"�� Loaded {len(self.labels_df)} cases from {split}/multi-plane")
        print(f"   Task: {task}")
        print(f"   Positive cases: {pos_count}")
        print(f"   Negative cases: {neg_count}")
        
        # Transforms
        if split == 'train':
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
    
    def __len__(self):
        return len(self.labels_df)
    
    def load_plane(self, case, plane):
        """Load a single plane for a given case"""
        volume_path = os.path.join(self.root_dir, self.split, plane, f'{case:04d}.npy')
        
        try:
            volume = np.load(volume_path)
        except:
            # Return zeros if file missing
            return torch.zeros(3, 224, 224)
        
        # Select slice
        if self.use_all_slices:
            slice_idx = np.random.randint(0, volume.shape[0])
        else:
            slice_idx = volume.shape[0] // 2
        
        slice_img = volume[slice_idx]
        
        # Normalize
        slice_img = (slice_img - slice_img.min()) / (slice_img.max() - slice_img.min() + 1e-8)
        slice_img = (slice_img * 255).astype(np.uint8)
        slice_img = Image.fromarray(slice_img).convert('RGB')
        
        # Transform
        if self.transform:
            slice_img = self.transform(slice_img)
        
        return slice_img
    
    def __getitem__(self, idx):
        case = self.labels_df.iloc[idx]['case']
        label = self.labels_df.iloc[idx]['label']
        
        # Load all three planes
        sagittal = self.load_plane(case, 'sagittal')
        coronal = self.load_plane(case, 'coronal')
        axial = self.load_plane(case, 'axial')
        
        return {
            'sagittal': sagittal,
            'coronal': coronal,
            'axial': axial,
            'label': torch.tensor(label, dtype=torch.float32)
        }
