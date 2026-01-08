import gradio as gr
import torch
import numpy as np
from PIL import Image
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.model import ResNetDeiTFusion
import torchvision.transforms as transforms

# Load model
print("Loading model...")
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = ResNetDeiTFusion(num_classes=1, fusion_type='concat', pretrained=False)
checkpoint = torch.load('outputs/models/best_model.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()
print("Model loaded successfully!")

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict_image(image):
    """Make prediction on uploaded image"""
    if image is None:
        return "Please upload an image", None
    
    # Transform
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        output = model(img_tensor)
        prob = torch.sigmoid(output).item()
    
    # Results
    prediction = "🔴 Abnormal" if prob > 0.5 else "🟢 Normal"
    confidence = prob if prob > 0.5 else (1 - prob)
    
    result_text = f"""
    ## Prediction Results
    
    **Diagnosis:** {prediction}
    
    **Confidence:** {confidence:.1%}
    
    **Abnormality Probability:** {prob:.3f}
    
    ---
    
    ### Interpretation:
    """
    
    if prob > 0.8:
        result_text += "⚠️ High confidence of abnormality detected. Recommend radiologist review."
    elif prob > 0.5:
        result_text += "⚠️ Possible abnormality detected. Further examination recommended."
    elif prob > 0.3:
        result_text += "⚡ Borderline result. Consider additional imaging or expert review."
    else:
        result_text += "✅ Low probability of abnormality. Appears normal."
    
    result_text += f"\n\n**Model AUC:** 91.37%"
    
    # Create probability chart
    labels = ['Normal', 'Abnormal']
    values = [1-prob, prob]
    
    return result_text, {"Normal": 1-prob, "Abnormal": prob}

# Create interface
with gr.Blocks(title="Knee MRI Abnormality Detection", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🦵 Knee MRI Abnormality Detection
    
    AI-powered knee MRI analysis using CNN-Transformer fusion model.
    
    **Model Performance:** 91.37% AUC | 84.2% Accuracy
    """)
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Upload Knee MRI Image")
            submit_btn = gr.Button("Analyze MRI", variant="primary", size="lg")
            gr.Markdown("""
            ### Instructions:
            1. Upload a sagittal plane knee MRI image
            2. Click "Analyze MRI"
            3. View prediction results
            
            ⚠️ **Disclaimer:** This is a research tool. Always consult a qualified radiologist for medical diagnosis.
            """)
        
        with gr.Column():
            result_text = gr.Markdown(label="Results")
            result_plot = gr.Label(label="Probability Distribution", num_top_classes=2)
    
    # Examples
    gr.Markdown("### Example Images")
    gr.Markdown("*Note: Add example images here if available*")
    
    # Button action
    submit_btn.click(
        fn=predict_image,
        inputs=image_input,
        outputs=[result_text, result_plot]
    )
    
    gr.Markdown("""
    ---
    ### About the Model
    
    - **Architecture:** ResNet50 + DeiT-Tiny Fusion
    - **Training Data:** MRNet v1.0 (1,130 training cases)
    - **Performance:** 91.37% AUC, 84.2% Accuracy
    - **Sensitivity:** ~94% (catches most abnormalities)
    
    **Author:** Yuvraj Pratap Singh | **Date:** October 2025
    """)

if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)
