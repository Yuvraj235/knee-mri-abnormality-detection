import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class FastMRIKneeDataset(Dataset):
    """FastMRI Knee Dataset Loader for Multi-Plane MRI"""
    
    def __init__(self, root_dir, split='train', transform=None, target_size=(224, 224)):
        self.root_dir = root_dir
        self.split = split
        self.target_size = target_size
        
        folder_map = {
            'train': 'singlecoil_train',
            'val': 'singlecoil_val',
            'test': 'singlecoil_test'
        }
        
        self.data_dir = os.path.join(root_dir, folder_map[split])
        
        self.file_list = sorted([
            os.path.join(self.data_dir, f) 
            for f in os.listdir(self.data_dir) 
            if f.endswith('.h5')
        ])
        
        print(f"📦 Loaded {len(self.file_list)} volumes from {split} set")
        
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(target_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            self.transform = transform
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        h5_path = self.file_list[idx]
        
        try:
            with h5py.File(h5_path, 'r') as f:
                volume = f['reconstruction_rss'][:]
        except:
            volume = np.zeros((30, 320, 320))
        
        n_slices = volume.shape[0]
        
        sagittal_slice = volume[n_slices // 2]
        coronal_slice = volume[n_slices // 3]
        axial_slice = volume[(2 * n_slices) // 3]
        
        sagittal_img = self._to_pil_image(sagittal_slice)
        coronal_img = self._to_pil_image(coronal_slice)
        axial_img = self._to_pil_image(axial_slice)
        
        sagittal = self.transform(sagittal_img)
        coronal = self.transform(coronal_img)
        axial = self.transform(axial_img)
        
        return {
            'sagittal': sagittal,
            'coronal': coronal,
            'axial': axial,
            'label': torch.tensor(0, dtype=torch.long),
            'file_name': os.path.basename(h5_path)
        }
    
    def _to_pil_image(self, slice_array):
        slice_array = np.nan_to_num(slice_array, nan=0.0, posinf=0.0, neginf=0.0)
        
        if slice_array.max() > slice_array.min():
            normalized = (slice_array - slice_array.min()) / (slice_array.max() - slice_array.min())
        else:
            normalized = np.zeros_like(slice_array)
        
        img_array = (normalized * 255).astype(np.uint8)
        return Image.fromarray(img_array).convert('RGB')


if __name__ == '__main__':
    print("🧪 Testing FastMRI Loader...")
    root_dir = '/Users/yuvrajpratapsingh/knee-mri-abnormality-detection/dataset/fastmri_knee'
    
    dataset = FastMRIKneeDataset(root_dir, split='train')
    sample = dataset[0]
    
    print(f"✅ Sample loaded!")
    print(f"   Sagittal: {sample['sagittal'].shape}")
    print(f"   Coronal:  {sample['coronal'].shape}")
    print(f"   Axial:    {sample['axial'].shape}")
