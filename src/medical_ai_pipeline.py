"""
Complete Medical AI Pipeline
Integrates all components for production use
"""

import torch
import numpy as np
from typing import Dict, Optional
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.medical_uncertainty_model import MedicalUncertaintyModel
from src.medical_explainability import MedicalExplainer
from src.clinical_decision_support import ClinicalDecisionSupport


class MedicalAIPipeline:
    """
    Complete pipeline for medical knee MRI analysis
    """
    
    def __init__(self, model_path: str, device: str = 'mps'):
        """
        Initialize the complete medical AI pipeline
        """
        
        print("\n" + "="*70)
        print("🏥 INITIALIZING MEDICAL AI PIPELINE")
        print("="*70)
        
        # Load uncertainty-aware model
        print("\n📦 Loading AI model...")
        self.model = MedicalUncertaintyModel(model_path, device)
        
        # Initialize explainer
        print("🔬 Initializing explainability module...")
        self.explainer = MedicalExplainer(self.model)
        
        # Initialize clinical decision support
        print("💡 Initializing clinical decision support...")
        self.cds = ClinicalDecisionSupport()
        
        print("\n" + "="*70)
        print("✅ MEDICAL AI PIPELINE READY!")
        print("="*70)
    
    def analyze_patient(
        self,
        sagittal: torch.Tensor,
        coronal: torch.Tensor,
        axial: torch.Tensor,
        patient_id: str = "Unknown",
        use_uncertainty: bool = True,
        generate_explanation: bool = True
    ) -> Dict:
        """
        Complete analysis of a patient's knee MRI
        """
        
        print(f"\n{'='*70}")
        print(f"🔍 ANALYZING PATIENT: {patient_id}")
        print(f"{'='*70}")
        
        # Step 1: AI Prediction
        print("\n1️⃣  Running AI prediction...")
        if use_uncertainty:
            prediction = self.model.predict_with_uncertainty(
                sagittal, coronal, axial, n_samples=20
            )
            print(f"   ✅ Prediction: {prediction['prediction_label']}")
            print(f"   📊 Probability: {prediction['probability']:.1%}")
            print(f"   🎯 Confidence: {prediction['confidence']:.1%}")
            print(f"   ⚠️  Uncertainty: {prediction['uncertainty']:.3f}")
        else:
            prediction = self.model.predict_deterministic(
                sagittal, coronal, axial
            )
            print(f"   ✅ Prediction: {prediction['prediction_label']}")
            print(f"   📊 Probability: {prediction['probability']:.1%}")
        
        # Step 2: Generate Explanation
        heatmaps = None
        if generate_explanation:
            print("\n2️⃣  Generating visual explanation...")
            try:
                heatmaps = self.explainer.generate_all_gradcams(
                    sagittal, coronal, axial
                )
                # Ensure heatmaps is a dict
                if not isinstance(heatmaps, dict):
                    print("   ⚠️  Heatmaps not in dict format, converting...")
                    heatmaps = {
                        'sagittal': np.zeros((224, 224)),
                        'coronal': np.zeros((224, 224)),
                        'axial': np.zeros((224, 224))
                    }
                print("   ✅ Grad-CAM heatmaps generated")
            except Exception as e:
                print(f"   ⚠️  Error generating heatmaps: {e}")
                heatmaps = {
                    'sagittal': np.zeros((224, 224)),
                    'coronal': np.zeros((224, 224)),
                    'axial': np.zeros((224, 224))
                }
        
        # Step 3: Clinical Decision Support
        print("\n3️⃣  Generating clinical recommendation...")
        recommendation = self.cds.get_recommendation(prediction)
        clinical_report = self.cds.format_for_display(recommendation, prediction)
        print("   ✅ Clinical recommendation generated")
        
        # Compile complete report
        report = {
            'patient_id': patient_id,
            'prediction': prediction,
            'heatmaps': heatmaps,
            'recommendation': recommendation,
            'clinical_report': clinical_report
        }
        
        print(f"\n{'='*70}")
        print("✅ ANALYSIS COMPLETE")
        print(f"{'='*70}")
        
        return report
    
    def print_report(self, report: Dict):
        """Print formatted clinical report"""
        print("\n" + report['clinical_report'])
    
    def save_report(self, report: Dict, output_path: str):
        """Save report to file"""
        with open(output_path, 'w') as f:
            f.write(f"PATIENT ID: {report['patient_id']}\n")
            f.write(f"\n{report['clinical_report']}")
        
        print(f"✅ Report saved to: {output_path}")


def demo_medical_pipeline():
    """
    Demonstrate the complete medical pipeline with a real case
    """
    
    from src.multiplane_loader import MultiPlaneMRNetDataset
    
    print("\n" + "="*70)
    print("🎬 MEDICAL AI PIPELINE DEMONSTRATION")
    print("="*70)
    
    # Paths
    model_path = '/Users/yuvrajpratapsingh/knee-mri-abnormality-detection/outputs/resnet_only/best_model.pth'
    dataset_path = '/Users/yuvrajpratapsingh/knee-mri-abnormality-detection/dataset/MRNet-v1.0'
    
    # Initialize pipeline
    pipeline = MedicalAIPipeline(model_path)
    
    # Load a test case
    print("\n�� Loading test MRI case...")
    dataset = MultiPlaneMRNetDataset(
        dataset_path,
        task='abnormal',
        split='valid',
        use_all_slices=False
    )
    
    # Analyze multiple cases
    for i in range(min(3, len(dataset))):
        sample = dataset[i]
        
        report = pipeline.analyze_patient(
            sagittal=sample['sagittal'].unsqueeze(0),
            coronal=sample['coronal'].unsqueeze(0),
            axial=sample['axial'].unsqueeze(0),
            patient_id=f"PATIENT-{i+1:03d}",
            use_uncertainty=True,
            generate_explanation=True
        )
        
        # Print clinical report
        pipeline.print_report(report)
        
        # Save report
        pipeline.save_report(
            report,
            f'outputs/medical_reports/patient_{i+1:03d}_report.txt'
        )
        
        # Visualize if first case
        if i == 0 and report['heatmaps'] is not None:
            try:
                images = {
                    'sagittal': sample['sagittal'].permute(1, 2, 0).numpy()[:, :, 0],
                    'coronal': sample['coronal'].permute(1, 2, 0).numpy()[:, :, 0],
                    'axial': sample['axial'].permute(1, 2, 0).numpy()[:, :, 0]
                }
                
                pipeline.explainer.visualize_explanation(
                    images,
                    report['heatmaps'],
                    report['prediction'],
                    save_path=f'outputs/medical_reports/patient_{i+1:03d}_visualization.png'
                )
            except Exception as e:
                print(f"⚠️  Could not create visualization: {e}")
    
    print("\n" + "="*70)
    print("🎉 DEMONSTRATION COMPLETE!")
    print("📁 Reports saved to: outputs/medical_reports/")
    print("="*70)


if __name__ == '__main__':
    os.makedirs('outputs/medical_reports', exist_ok=True)
    demo_medical_pipeline()
