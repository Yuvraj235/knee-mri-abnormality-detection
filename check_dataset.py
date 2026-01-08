import os
import pandas as pd
import numpy as np

mrnet_path = '/Users/yuvrajpratapsingh/knee-mri-abnormality-detection/dataset/MRNet-v1.0'
plane = 'sagittal'
task = 'abnormal'

for split in ['train', 'valid']:
    print(f"\n{'='*60}")
    print(f"Checking {split} set...")
    print(f"{'='*60}")
    
    # Load labels
    label_file = f'{split}-{task}.csv'
    label_path = os.path.join(mrnet_path, label_file)
    
    if not os.path.exists(label_path):
        label_file = f'{split}_{task}.csv'
        label_path = os.path.join(mrnet_path, label_file)
    
    df = pd.read_csv(label_path, names=['case', 'label'], header=None)
    print(f"CSV has {len(df)} cases")
    
    # Check which files exist
    missing = []
    for idx, row in df.iterrows():
        case = row['case']
        volume_path = os.path.join(mrnet_path, split, plane, f'{case}.npy')
        if not os.path.exists(volume_path):
            missing.append(case)
    
    if missing:
        print(f"⚠️  Missing {len(missing)} files: {missing[:10]}")
    else:
        print(f"✅ All files present!")
