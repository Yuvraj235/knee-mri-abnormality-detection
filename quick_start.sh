#!/bin/bash

echo "🚀 Knee MRI Abnormality Detection - Quick Start"
echo "==============================================="

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Creating one..."
    python3 -m venv venv
fi

# Activate venv
source venv/bin/activate

# Check if requirements are installed
echo -e "\n📦 Checking dependencies..."
python -c "import torch; import transformers; import timm" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  Installing dependencies..."
    pip install -q -r requirements.txt
fi

# Fix SSL issue
echo -e "\n🔧 Checking model weights..."
if [ ! -f ~/.cache/torch/hub/checkpoints/resnet50-11ad3fa6.pth ]; then
    echo "📥 Downloading ResNet50 weights..."
    mkdir -p ~/.cache/torch/hub/checkpoints
    curl -L -k -o ~/.cache/torch/hub/checkpoints/resnet50-11ad3fa6.pth \
        https://download.pytorch.org/models/resnet50-11ad3fa6.pth
fi

# Run tests
echo -e "\n🧪 Running tests..."
./run_all_tests.sh

# Start training
echo -e "\n🏋️  Starting training..."
read -p "Do you want to start training now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python src/train.py
else
    echo "ℹ️  To start training later, run: python src/train.py"
fi
