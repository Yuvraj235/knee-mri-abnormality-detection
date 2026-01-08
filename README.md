# 🦵 Knee MRI Abnormality Detection

CNN-Transformer fusion model for detecting abnormalities in knee MRI scans using the MRNet dataset.

## 🏗️ Architecture

- **CNN Branch**: ResNet50 (25M parameters)
- **Transformer Branch**: DeiT-Tiny (5.7M parameters)
- **Fusion Method**: Cross-attention mechanism
- **Total Parameters**: ~31M
- **Hardware**: Optimized for Mac M1/M2/M3 with MPS

## 📂 Project Structure
```
knee-mri-abnormality-detection/
├── dataset/MRNet-v1.0/          # MRNet dataset
│   ├── train/
│   │   ├── sagittal/
│   │   ├── coronal/
│   │   └── axial/
│   └── *.csv (labels)
│
├── src/
│   ├── data_loader.py           # Dataset loader
│   ├── model.py                 # CNN-Transformer fusion model
│   ├── train.py                 # Training script
│   ├── evaluate.py              # Evaluation script
│   ├── predict.py               # Inference script
│   └── explainability.py        # Grad-CAM & attention viz
│
├── outputs/
│   ├── models/                  # Saved checkpoints
│   ├── plots/                   # Training curves, visualizations
│   └── predictions/             # Prediction results
│
├── configs/                     # Configuration files
├── notebooks/                   # Jupyter notebooks
└── tests/                       # Unit tests
```

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Fix SSL certificates (if needed)
curl -L -k -o ~/.cache/torch/hub/checkpoints/resnet50-11ad3fa6.pth \
  https://download.pytorch.org/models/resnet50-11ad3fa6.pth
```

### 2. Verify Setup
```bash
# Test data loader
python src/data_loader.py

# Test model
python src/model.py
```

### 3. Train Model
```bash
# Start training (50 epochs, ~3-4 hours on Mac M1/M2)
python src/train.py
```

### 4. Evaluate Model
```bash
# Evaluate on validation set
python src/evaluate.py
```

### 5. Make Predictions
```bash
# Run inference on new data
python src/predict.py
```

## 📊 Expected Results

| Metric | Expected Range |
|--------|---------------|
| **AUC** | 0.80 - 0.88 |
| **Accuracy** | 0.75 - 0.85 |
| **F1 Score** | 0.70 - 0.82 |
| **Training Time** | 3-5 min/epoch |

## 🎯 Dataset

**MRNet v1.0** - Stanford ML Group
- Training: 1,130 knee MRI exams
- Validation: 120 exams
- Tasks: Abnormal, ACL tear, Meniscal tear
- Format: 3D volumes (.npy), 256×256 slices

## 📖 Usage Examples

### Training with Custom Config
```python
CONFIG = {
    'plane': 'sagittal',        # or 'coronal', 'axial'
    'task': 'abnormal',         # or 'acl', 'meniscus'
    'batch_size': 8,
    'num_epochs': 50,
    'learning_rate': 1e-4,
    'fusion_type': 'attention'  # or 'concat', 'add'
}
```

### Loading Trained Model
```python
from src.model import ResNetDeiTFusion
import torch

model = ResNetDeiTFusion(num_classes=1, fusion_type='attention')
checkpoint = torch.load('outputs/models/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
```

## 🔍 Explainability

The model includes attention visualization and Grad-CAM for interpretability:
```bash
python src/explainability.py
```

## 📝 Citation
```bibtex
@article{mrnet2018,
  title={Deep-learning-assisted diagnosis for knee magnetic resonance imaging},
  author={Bien, Nicholas and others},
  journal={PLoS medicine},
  year={2018}
}
```

## 📄 License

This project is for educational purposes.

## 👨‍💻 Author

Yuvraj Pratap Singh

---

**Note**: This model achieves competitive performance with the original MRNet paper while being optimized for Mac hardware.
