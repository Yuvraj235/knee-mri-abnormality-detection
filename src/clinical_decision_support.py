"""
Clinical Decision Support System
Provides actionable recommendations for doctors
"""

from typing import Dict
from dataclasses import dataclass


@dataclass
class ClinicalRecommendation:
    """Structure for clinical recommendations"""
    action: str
    urgency: str  # 'routine', 'elevated', 'urgent'
    rationale: str
    follow_up: str


class ClinicalDecisionSupport:
    """
    Translates AI predictions into clinical recommendations
    """
    
    @staticmethod
    def get_recommendation(prediction: Dict[str, any]) -> ClinicalRecommendation:
        """
        Generate clinical recommendation based on prediction
        
        Args:
            prediction: Dictionary with prediction, probability, confidence, uncertainty
        
        Returns:
            ClinicalRecommendation object
        """
        
        prob = prediction['probability']
        conf = prediction['confidence']
        unc = prediction.get('uncertainty', 0)
        pred_class = prediction['prediction_class']
        
        # HIGH CONFIDENCE ABNORMAL
        if pred_class == 1 and prob > 0.85 and conf > 0.80:
            return ClinicalRecommendation(
                action="Order comprehensive knee MRI with contrast",
                urgency="elevated",
                rationale=f"High confidence abnormality detected (Probability: {prob:.1%}, Confidence: {conf:.1%}). "
                         f"Model indicates structural abnormality with high certainty.",
                follow_up="Recommend orthopedic consultation within 1-2 weeks. "
                         "Consider ACL, meniscus, or cartilage evaluation."
            )
        
        # MODERATE CONFIDENCE ABNORMAL
        elif pred_class == 1 and prob > 0.65:
            return ClinicalRecommendation(
                action="Manual radiologist review recommended",
                urgency="routine",
                rationale=f"Possible abnormality detected (Probability: {prob:.1%}, Confidence: {conf:.1%}). "
                         f"AI suggests potential finding requiring expert verification.",
                follow_up="Radiologist review within 3-5 business days. "
                         "Compare with prior imaging if available."
            )
        
        # LOW CONFIDENCE / UNCERTAIN
        elif unc > 0.15:
            return ClinicalRecommendation(
                action="Mandatory expert radiologist review",
                urgency="routine",
                rationale=f"AI model uncertain about diagnosis (Uncertainty: {unc:.3f}). "
                         f"Multiple factors may be contributing to ambiguous presentation.",
                follow_up="Comprehensive radiologist evaluation required. "
                         "Consider additional views or modalities if indicated."
            )
        
        # HIGH CONFIDENCE NORMAL
        elif pred_class == 0 and prob < 0.15 and conf > 0.80:
            return ClinicalRecommendation(
                action="Routine follow-up",
                urgency="routine",
                rationale=f"High confidence normal scan (Probability of abnormality: {prob:.1%}). "
                         f"No significant structural abnormalities detected by AI.",
                follow_up="Continue routine monitoring. "
                         "Repeat imaging only if clinically indicated."
            )
        
        # MODERATE CONFIDENCE NORMAL
        else:
            return ClinicalRecommendation(
                action="Consider radiologist review if clinical suspicion high",
                urgency="routine",
                rationale=f"AI assessment suggests likely normal (Probability of abnormality: {prob:.1%}). "
                         f"However, clinical correlation always required.",
                follow_up="If patient symptoms persist or worsen, consider follow-up imaging "
                         "or specialist consultation regardless of AI assessment."
            )
    
    @staticmethod
    def format_for_display(recommendation: ClinicalRecommendation, prediction: Dict) -> str:
        """
        Format recommendation for doctor-friendly display
        """
        
        urgency_emoji = {
            'routine': '🟢',
            'elevated': '🟡',
            'urgent': '🔴'
        }
        
        report = f"""
╔══════════════════════════════════════════════════════════════╗
║                  CLINICAL DECISION SUPPORT                   ║
╠══════════════════════════════════════════════════════════════╣

📋 AI ANALYSIS:
   Prediction:    {prediction['prediction_label']}
   Probability:   {prediction['probability']:.1%}
   Confidence:    {prediction['confidence']:.1%}
   Uncertainty:   {prediction.get('uncertainty', 0):.3f}

{urgency_emoji[recommendation.urgency]} RECOMMENDED ACTION ({recommendation.urgency.upper()}):
   {recommendation.action}

💡 CLINICAL RATIONALE:
   {recommendation.rationale}

📅 FOLLOW-UP:
   {recommendation.follow_up}

⚠️  DISCLAIMER:
   This AI system is intended as a diagnostic aid only.
   Final diagnosis must be made by qualified healthcare
   professionals based on clinical judgment and complete
   patient evaluation.

╚══════════════════════════════════════════════════════════════╝
"""
        return report


def test_clinical_support():
    """Test clinical decision support"""
    
    print("\n" + "="*70)
    print("🏥 TESTING CLINICAL DECISION SUPPORT")
    print("="*70)
    
    cds = ClinicalDecisionSupport()
    
    # Test cases
    test_cases = [
        {
            'name': 'High Confidence Abnormal',
            'prediction': {
                'prediction_class': 1,
                'prediction_label': 'ABNORMAL',
                'probability': 0.92,
                'confidence': 0.88,
                'uncertainty': 0.05
            }
        },
        {
            'name': 'Moderate Confidence Abnormal',
            'prediction': {
                'prediction_class': 1,
                'prediction_label': 'ABNORMAL',
                'probability': 0.72,
                'confidence': 0.65,
                'uncertainty': 0.12
            }
        },
        {
            'name': 'Uncertain Case',
            'prediction': {
                'prediction_class': 1,
                'prediction_label': 'ABNORMAL',
                'probability': 0.58,
                'confidence': 0.45,
                'uncertainty': 0.22
            }
        },
        {
            'name': 'High Confidence Normal',
            'prediction': {
                'prediction_class': 0,
                'prediction_label': 'NORMAL',
                'probability': 0.08,
                'confidence': 0.92,
                'uncertainty': 0.03
            }
        }
    ]
    
    for test_case in test_cases:
        print(f"\n{'='*70}")
        print(f"TEST CASE: {test_case['name']}")
        print(f"{'='*70}")
        
        recommendation = cds.get_recommendation(test_case['prediction'])
        report = cds.format_for_display(recommendation, test_case['prediction'])
        
        print(report)
    
    print("\n" + "="*70)
    print("✅ CLINICAL DECISION SUPPORT READY!")
    print("="*70)


if __name__ == '__main__':
    test_clinical_support()
