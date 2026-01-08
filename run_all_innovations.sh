#!/bin/bash

echo "=========================================="
echo "🚀 KNEE MRI INNOVATION PIPELINE"
echo "=========================================="

# Step 1: Threshold Optimization (Quick Win)
echo ""
echo "Step 1: Running threshold optimization..."
python src/optimize_threshold.py

# Wait for user
echo ""
echo "Press Enter to continue to multi-plane training..."
read

# Step 2: Multi-Plane Training
echo ""
echo "Step 2: Training multi-plane fusion model..."
python src/train_multiplane.py

echo ""
echo "=========================================="
echo "✅ ALL INNOVATIONS COMPLETE!"
echo "=========================================="
echo ""
echo "Results saved in:"
echo "  - outputs/improved_analysis/threshold_optimization.png"
echo "  - outputs/models/best_model_multiplane.pth"
echo "  - outputs/plots/training_multiplane.png"
echo ""
