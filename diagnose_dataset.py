import os
import pandas as pd

mrnet_path = '/Users/yuvrajpratapsingh/knee-mri-abnormality-detection/dataset/MRNet-v1.0'
plane = 'sagittal'

print("\n" + "="*60)
print("Checking actual files in train directory...")
print("="*60)

train_dir = os.path.join(mrnet_path, 'train', plane)
if os.path.exists(train_dir):
    files = sorted([f for f in os.listdir(train_dir) if f.endswith('.npy')])
    print(f"Found {len(files)} .npy files")
    print(f"First 10 files: {files[:10]}")
    print(f"Last 10 files: {files[-10:]}")
    
    # Extract case numbers
    case_numbers = sorted([int(f.replace('.npy', '')) for f in files])
    print(f"\nCase number range: {min(case_numbers)} to {max(case_numbers)}")
else:
    print(f"Directory doesn't exist: {train_dir}")

print("\n" + "="*60)
print("Checking valid directory...")
print("="*60)

valid_dir = os.path.join(mrnet_path, 'valid', plane)
if os.path.exists(valid_dir):
    files = sorted([f for f in os.listdir(valid_dir) if f.endswith('.npy')])
    print(f"Found {len(files)} .npy files")
    print(f"First 10 files: {files[:10]}")
    
    case_numbers = sorted([int(f.replace('.npy', '')) for f in files])
    print(f"Case number range: {min(case_numbers)} to {max(case_numbers)}")
else:
    print(f"Directory doesn't exist: {valid_dir}")

print("\n" + "="*60)
print("Checking CSV format...")
print("="*60)

for split in ['train', 'valid']:
    csv_path = os.path.join(mrnet_path, f'{split}-abnormal.csv')
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, names=['case', 'label'], header=None)
        print(f"\n{split} CSV:")
        print(f"  Total cases: {len(df)}")
        print(f"  First 10 cases: {df['case'].head(10).tolist()}")
        print(f"  Data types: {df.dtypes.to_dict()}")
