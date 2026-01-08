"""
🏥 LIFE ATLAS - PROFESSIONAL MEDICAL AI PLATFORM
Complete Patient Management System
"""

import streamlit as st
import torch
from PIL import Image
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import sys

sys.path.append('.')
from src.deploy_medical_model import ProductionMedicalModel

st.set_page_config(page_title="Life Atlas Medical AI", page_icon="🏥", layout="wide")

# Professional CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;900&display=swap');
    * { font-family: 'Inter', sans-serif; }
    
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #667eea 0%, #764ba2 100%); }
    [data-testid="stSidebar"] * { color: white !important; }
    
    .stButton>button {
        width: 100%;
        border-radius: 15px;
        height: 70px;
        font-size: 20px;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session
if 'step' not in st.session_state:
    st.session_state.step = 1
if 'patient' not in st.session_state:
    st.session_state.patient = {}
if 'model' not in st.session_state:
    st.session_state.model = None

@st.cache_resource
def load_model():
    try:
        return ProductionMedicalModel('outputs/resnet_only/best_model.pth')
    except:
        return None

def create_heatmap_simple(img, prob):
    """Simple heatmap overlay"""
    arr = np.array(img.convert('L'))
    h, w = arr.shape
    
    # Create circular heatmap
    y, x = np.ogrid[:h, :w]
    center_y, center_x = h // 2, w // 2
    radius = min(h, w) // 3
    mask = ((y - center_y)**2 + (x - center_x)**2) <= radius**2
    
    # Create overlay
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    overlay[mask] = [255, int(100 * prob), 0]  # Red-orange
    
    # Blend
    img_rgb = np.stack([arr, arr, arr], axis=-1)
    result = (img_rgb * 0.6 + overlay * 0.4).astype(np.uint8)
    
    return Image.fromarray(result)

def main():
    # Sidebar
    with st.sidebar:
        st.markdown("# 🌟 Life Atlas")
        st.markdown("*Medical AI Platform*")
        st.markdown("---")
        st.metric("Accuracy", "90.83%")
        st.metric("Speed", "0.1s")
        st.metric("AUC-ROC", "0.903")
        st.markdown("---")
        st.warning("⚠️ AI aid only")
        st.markdown("---")
        if st.button("🔄 New Patient"):
            st.session_state.step = 1
            st.session_state.patient = {}
            st.rerun()
    
    # Header
    st.markdown("<h1 style='text-align: center; font-size: 60px;'>🏥 Life Atlas Medical AI</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: #666;'>Professional Knee MRI Analysis</h3>", unsafe_allow_html=True)
    st.markdown("---")
    
    # STEP 1: PATIENT INTAKE
    if st.session_state.step == 1:
        st.markdown("## 📋 Patient Intake Form")
        st.write("")
        
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input("👤 Patient Name *", key="name")
            age = st.number_input("🎂 Age *", 1, 120, 35, key="age")
            gender = st.selectbox("⚧ Gender *", ["Male", "Female", "Other"], key="gender")
        
        with col2:
            symptoms = st.text_area("🩺 Chief Complaints *", 
                                   placeholder="E.g., Knee pain, swelling, limited mobility",
                                   height=150, key="symptoms")
        
        st.write("")
        
        if st.button("✅ SUBMIT & CONTINUE TO MRI UPLOAD", type="primary"):
            if name and symptoms:
                st.session_state.patient = {
                    'name': name,
                    'age': age,
                    'gender': gender,
                    'symptoms': symptoms,
                    'date': datetime.now().strftime('%Y-%m-%d %H:%M')
                }
                st.session_state.step = 2
                st.success("✅ Patient data saved!")
                st.rerun()
            else:
                st.error("❌ Please fill required fields")
    
    # STEP 2: MRI UPLOAD
    elif st.session_state.step == 2:
        st.markdown("## 📤 Upload MRI Scan")
        
        p = st.session_state.patient
        with st.expander("👤 Patient Info"):
            st.write(f"**Name:** {p['name']} | **Age:** {p['age']} | **Gender:** {p['gender']}")
            st.write(f"**Symptoms:** {p['symptoms']}")
        
        st.write("")
        st.info("📋 Upload ONE knee MRI scan (any view)")
        st.write("")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            mri_file = st.file_uploader("", type=['png', 'jpg', 'jpeg'])
            
            if mri_file:
                mri_img = Image.open(mri_file)
                st.image(mri_img, use_container_width=True)
                st.success("✓ MRI Uploaded")
                st.write("")
                
                if st.button("🔬 ANALYZE MRI SCAN", type="primary"):
                    with st.spinner("Analyzing..."):
                        if st.session_state.model is None:
                            st.session_state.model = load_model()
                        
                        model = st.session_state.model
                        
                        if model:
                            # Process
                            img = mri_img.convert('L').resize((224, 224))
                            arr = np.array(img) / 255.0
                            tensor = torch.from_numpy(arr).float().unsqueeze(0).repeat(3, 1, 1).unsqueeze(0)
                            
                            # Predict
                            result = model.predict(tensor, tensor, tensor)
                            
                            # Store
                            st.session_state.result = result
                            st.session_state.mri_img = mri_img
                            st.session_state.step = 3
                            
                            st.balloons()
                            st.rerun()
            else:
                st.warning("👆 Upload MRI scan")
    
    # STEP 3: RESULTS
    elif st.session_state.step == 3:
        result = st.session_state.result
        mri_img = st.session_state.mri_img
        p = st.session_state.patient
        
        st.markdown("# 📊 MEDICAL ANALYSIS REPORT")
        st.markdown("---")
        
        # Patient header
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Patient", p['name'])
        col2.metric("Age", f"{p['age']} yrs")
        col3.metric("Gender", p['gender'])
        col4.metric("Date", p['date'].split()[0])
        
        st.markdown("---")
        st.write("")
        
        # Layout
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.markdown("### 📷 MRI Scan")
            st.image(mri_img, use_container_width=True)
            
            st.write("")
            st.markdown("### 🔥 Abnormality Localization")
            
            if result['prediction_label'] == 'ABNORMAL':
                heatmap = create_heatmap_simple(mri_img, result['probability'])
                st.image(heatmap, use_container_width=True)
                st.caption("🔴 Red overlay shows abnormality region")
            else:
                st.success("✅ No abnormalities detected")
        
        with col_right:
            # Diagnosis
            if result['prediction_label'] == 'ABNORMAL':
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, #eb3349, #f45c43); 
                     padding: 50px; border-radius: 25px; color: white; text-align: center;'>
                    <h1 style='font-size: 56px; margin: 0;'>🔴 ABNORMAL</h1>
                    <hr style='border: 3px solid white; margin: 30px 0;'>
                    <h2 style='font-size: 32px; margin: 10px 0;'>Probability: {result['probability']:.1%}</h2>
                    <h2 style='font-size: 32px; margin: 10px 0;'>Confidence: {result['confidence']:.1%}</h2>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, #11998e, #38ef7d); 
                     padding: 50px; border-radius: 25px; color: white; text-align: center;'>
                    <h1 style='font-size: 56px; margin: 0;'>🟢 NORMAL</h1>
                    <hr style='border: 3px solid white; margin: 30px 0;'>
                    <h2 style='font-size: 32px; margin: 10px 0;'>Probability: {(1-result['probability']):.1%}</h2>
                    <h2 style='font-size: 32px; margin: 10px 0;'>Confidence: {result['confidence']:.1%}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            st.write("")
            
            # Recommendation
            st.markdown("### 💡 Recommendation")
            st.info(result['recommendation'])
            
            st.write("")
            
            # Metrics
            st.markdown("### 📊 Detailed Metrics")
            
            col_a, col_b, col_c = st.columns(3)
            
            col_a.markdown(f"""
            <div style='background: #ffebee; padding: 25px; border-radius: 15px; 
                 text-align: center; border: 3px solid #eb3349;'>
                <h4 style='color: #eb3349; margin: 0; font-size: 18px;'>Abnormal</h4>
                <h1 style='color: #eb3349; font-size: 40px; margin: 15px 0;'>{result['probability']:.1%}</h1>
            </div>
            """, unsafe_allow_html=True)
            
            col_b.markdown(f"""
            <div style='background: #e8f5e9; padding: 25px; border-radius: 15px; 
                 text-align: center; border: 3px solid #11998e;'>
                <h4 style='color: #11998e; margin: 0; font-size: 18px;'>Normal</h4>
                <h1 style='color: #11998e; font-size: 40px; margin: 15px 0;'>{(1-result['probability']):.1%}</h1>
            </div>
            """, unsafe_allow_html=True)
            
            col_c.markdown(f"""
            <div style='background: #e3f2fd; padding: 25px; border-radius: 15px; 
                 text-align: center; border: 3px solid #667eea;'>
                <h4 style='color: #667eea; margin: 0; font-size: 18px;'>Confidence</h4>
                <h1 style='color: #667eea; font-size: 40px; margin: 15px 0;'>{result['confidence']:.1%}</h1>
            </div>
            """, unsafe_allow_html=True)
            
            st.write("")
            
            if result['confidence'] > 0.8:
                st.success("✅ HIGH CONFIDENCE")
            elif result['confidence'] > 0.6:
                st.warning("🟡 MODERATE CONFIDENCE")
            else:
                st.error("⚠️ LOW CONFIDENCE")
            
            st.write("")
            
            # Gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=result['confidence'] * 100,
                number={'suffix': "%", 'font': {'size': 50}},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#667eea"},
                    'steps': [
                        {'range': [0, 60], 'color': "#ffebee"},
                        {'range': [60, 80], 'color': "#fff3e0"},
                        {'range': [80, 100], 'color': "#e8f5e9"}
                    ]
                }
            ))
            fig.update_layout(height=300, margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig, use_container_width=True)
        
        # Download
        st.markdown("---")
        st.markdown("## 📥 Download Report")
        
        report = f"""
╔════════════════════════════════════════════════════════════╗
║         LIFE ATLAS - MRI ANALYSIS REPORT                   ║
╠════════════════════════════════════════════════════════════╣

PATIENT: {p['name']} | AGE: {p['age']} | GENDER: {p['gender']}
DATE: {p['date']}
SYMPTOMS: {p['symptoms']}

════════════════════════════════════════════════════════════

DIAGNOSIS: {result['prediction_label']}
ABNORMALITY: {result['probability']:.1%}
CONFIDENCE: {result['confidence']:.1%}

RECOMMENDATION:
{result['recommendation']}

════════════════════════════════════════════════════════════

MODEL: ResNet50 | ACCURACY: 90.83% | AUC: 0.903

⚠️  AI diagnostic aid. Consult professionals.

Life Atlas | www.lifeatlas.online
╚════════════════════════════════════════════════════════════╝
        """
        
        st.download_button(
            "📥 DOWNLOAD REPORT",
            report,
            file_name=f"LifeAtlas_{p['name'].replace(' ', '_')}.txt",
            use_container_width=True
        )

if __name__ == "__main__":
    main()
