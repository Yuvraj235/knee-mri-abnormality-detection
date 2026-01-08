#!/bin/bash

echo "🧪 Running All Tests"
echo "===================="

# Activate virtual environment
source venv/bin/activate

# Test 1: Data Loader
echo -e "\n1️⃣  Testing Data Loader..."
python src/data_loader.py
if [ $? -eq 0 ]; then
    echo "✅ Data loader test passed"
else
    echo "❌ Data loader test failed"
    exit 1
fi

# Test 2: Model
echo -e "\n2️⃣  Testing Model..."
python src/model.py
if [ $? -eq 0 ]; then
    echo "✅ Model test passed"
else
    echo "❌ Model test failed"
    exit 1
fi

# Test 3: Check dataset
echo -e "\n3️⃣  Checking Dataset..."
if [ -d "dataset/MRNet-v1.0/train/sagittal" ]; then
    num_files=$(ls dataset/MRNet-v1.0/train/sagittal/*.npy | wc -l)
    echo "✅ Found $num_files training files"
else
    echo "❌ Dataset not found"
    exit 1
fi

echo -e "\n🎉 All tests passed! Ready to train."
