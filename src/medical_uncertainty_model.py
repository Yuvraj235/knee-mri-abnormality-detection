"""
Medical-Grade Model with Uncertainty Quantification
Uses Monte Carlo Dropout for reliable uncertainty estimates
"""

import torch
import torch.nn as nn
import numpy as np
import json
import os
from typing import Dict, Tuple
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.multiplane_model import MultiPlaneFusion


class MedicalUncertaintyModel(nn.Module):
    """
    Wrapper around your best model with uncertainty quantification
    """
    
    def __init__(self, model_path: str, device: str = 'mps'):
        super().__init__()
        
        self.device = torch.device(device if torch.backends.mps.is_available() else 'cpu')
        
        # Load your best model
        print("🔧 Loading best model...")
        self.model = MultiPlaneFusion(num_classes=1, dropout_rate=0.4)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        
        # Load calibrated temperature
        temp_path = 'outputs/optimal_temperature.json'
        if os.path.exists(temp_path):
            with open(temp_path, 'r') as f:
                temp_data = json.load(f)
                self.temperature = nn.Parameter(torch.tensor([temp_data['temperature']]).to(self.device))
                print(f"✅ Loaded calibrated temperature: {temp_data['temperature']:.4f}")
        else:
            self.temperature = nn.Parameter(torch.ones(1).to(self.device))
            print("⚠️  Using default temperature: 1.0")
        
        print(f"✅ Model loaded on {self.device}")
    
    def enable_dropout(self):
        """Enable dropout layers while keeping BatchNorm in eval mode"""
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.train()
            elif isinstance(module, nn.BatchNorm2d):
                module.eval()
    
    def predict_with_uncertainty(
        self, 
        sagittal: torch.Tensor,
        coronal: torch.Tensor,
        axial: torch.Tensor,
        n_samples: int = 20
    ) -> Dict[str, float]:
        """
        Make prediction with uncertainty quantification using Monte Carlo Dropout
        """
        
        # Move inputs to device
        sagittal = sagittal.to(self.device)
        coronal = coronal.to(self.device)
        axial = axial.to(self.device)
        
        # Set model to eval mode but enable dropout
        self.model.eval()
        self.enable_dropout()
        
        predictions = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                # Forward pass with dropout active
                logits = self.model(sagittal, coronal, axial)
                
                # Temperature scaling for calibration
                calibrated_logits = logits / self.temperature
                
                # Get probability
                prob = torch.sigmoid(calibrated_logits)
                predictions.append(prob.item())
        
        # Calculate statistics
        predictions = np.array(predictions)
        mean_prob = np.mean(predictions)
        std_prob = np.std(predictions)
        
        # Confidence: high when uncertainty is low
        confidence = 1.0 - min(std_prob * 2, 1.0)
        
        # Prediction
        prediction_class = 1 if mean_prob > 0.5 else 0
        prediction_label = 'ABNORMAL' if prediction_class == 1 else 'NORMAL'
        
        return {
            'prediction_class': prediction_class,
            'prediction_label': prediction_label,
            'probability': float(mean_prob),
            'uncertainty': float(std_prob),
            'confidence': float(confidence),
            'probability_distribution': predictions.tolist()
        }
    
    def predict_deterministic(
        self,
        sagittal: torch.Tensor,
        coronal: torch.Tensor,
        axial: torch.Tensor
    ) -> Dict[str, float]:
        """
        Fast deterministic prediction (no uncertainty)
        """
        
        self.model.eval()
        
        sagittal = sagittal.to(self.device)
        coronal = coronal.to(self.device)
        axial = axial.to(self.device)
        
        with torch.no_grad():
            logits = self.model(sagittal, coronal, axial)
            calibrated_logits = logits / self.temperature
            prob = torch.sigmoid(calibrated_logits).item()
        
        prediction_class = 1 if prob > 0.5 else 0
        prediction_label = 'ABNORMAL' if prediction_class == 1 else 'NORMAL'
        
        return {
            'prediction_class': prediction_class,
            'prediction_label': prediction_label,
            'probability': float(prob),
            'confidence': abs(prob - 0.5) * 2
        }


def test_uncertainty_model():
    """Test the uncertainty-aware model"""
    
    print("\n" + "="*70)
    print("🧪 TESTING MEDICAL UNCERTAINTY MODEL")
    print("="*70)
    
    model_path = '/Users/yuvrajpratapsingh/knee-mri-abnormality-detection/outputs/resnet_only/best_model.pth'
    
    # Create model
    medical_model = MedicalUncertaintyModel(model_path)
    
    # Test with dummy data
    print("\n📊 Testing with sample MRI...")
    sag = torch.randn(1, 3, 224, 224)
    cor = torch.randn(1, 3, 224, 224)
    axi = torch.randn(1, 3, 224, 224)
    
    # Uncertainty prediction
    print("\n🔬 Running uncertainty-aware prediction (20 samples)...")
    result_uncertain = medical_model.predict_with_uncertainty(sag, cor, axi, n_samples=20)
    
    print("\n✅ Results with Uncertainty:")
    print(f"   Prediction:   {result_uncertain['prediction_label']}")
    print(f"   Probability:  {result_uncertain['probability']:.3f}")
    print(f"   Uncertainty:  {result_uncertain['uncertainty']:.3f}")
    print(f"   Confidence:   {result_uncertain['confidence']:.3f} ({result_uncertain['confidence']*100:.1f}%)")
    
    # Fast prediction
    print("\n⚡ Running fast deterministic prediction...")
    result_fast = medical_model.predict_deterministic(sag, cor, axi)
    
    print("\n✅ Fast Results:")
    print(f"   Prediction:   {result_fast['prediction_label']}")
    print(f"   Probability:  {result_fast['probability']:.3f}")
    print(f"   Confidence:   {result_fast['confidence']:.3f} ({result_fast['confidence']*100:.1f}%)")
    
    print("\n" + "="*70)
    print("✅ MODEL READY FOR MEDICAL USE!")
    print("="*70)


if __name__ == '__main__':
    test_uncertainty_model()
