#!/bin/bash

echo "=========================================="
echo "🚀 COMPLETE PIPELINE EXECUTION"
echo "=========================================="

# STEP 1: Multi-Plane Training
echo ""
echo "STEP 1: Training Multi-Plane Model (~20 minutes)"
echo "=========================================="
python src/train_multiplane_fixed.py

# Check if training succeeded
if [ ! -f "outputs/models/best_model_multiplane.pth" ]; then
    echo "❌ Training failed! Check errors above."
    exit 1
fi

echo ""
echo "✅ Training complete!"
echo ""

# STEP 2: Analyze Multi-Plane Model
echo "STEP 2: Analyzing Multi-Plane Model"
echo "=========================================="
python src/analyze_multiplane.py

echo ""
echo "✅ Analysis complete!"
echo ""

# STEP 3: Compare All Models
echo "STEP 3: Comparing All Models"
echo "=========================================="
python src/compare_all_models.py

echo ""
echo "✅ Comparison complete!"
echo ""

# STEP 4: Summary
echo "=========================================="
echo "🎉 ALL STEPS COMPLETE!"
echo "=========================================="
echo ""
echo "📁 Results saved in:"
echo "  ├─ outputs/models/best_model_multiplane.pth"
echo "  ├─ outputs/plots/training_multiplane.png"
echo "  ├─ outputs/multiplane_analysis/predictions_multiplane.csv"
echo "  ├─ outputs/final_comparison.png"
echo "  └─ outputs/improved_analysis/threshold_optimization.png"
echo ""
echo "🎯 Your project is now complete with:"
echo "  ✅ Baseline model analysis"
echo "  ✅ Single-plane improved model"
echo "  ✅ Multi-plane fusion model"
echo "  ✅ Threshold optimization"
echo "  ✅ Complete comparison"
echo ""
echo "=========================================="
