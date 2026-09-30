"""
Plant Disease Classification & Health Diagnosis - Streamlit Web Application
Enhanced with OpenCV Automatic Leaf Segmentation, Test-Time Augmentation (TTA),
Binary Health Diagnosis, and Out-of-Distribution Protection for Google Images.
"""

import os
import sys
import json

# Ensure local project directory is at position 0 of sys.path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import torchvision.transforms as T

from src.model import PlantDiseaseCNN, get_transfer_learning_model

try:
    from src.utils import DISEASE_INFO, clean_class_name, extract_leaf_region
except ImportError:
    # Direct fallback definitions to ensure failproof execution
    import cv2
    
    DISEASE_INFO = {
        "Pepper__bell___Bacterial_spot": {"crop": "Pepper (Bell)", "condition": "Bacterial Spot", "status": "Diseased", "description": "Small spots turning brown.", "prevention": "Crop rotation.", "treatment": "Copper bactericide."},
        "Pepper__bell___healthy": {"crop": "Pepper (Bell)", "condition": "Healthy Leaf", "status": "Healthy", "description": "Vibrant green leaf.", "prevention": "Good watering.", "treatment": "None needed."},
        "Potato___Early_blight": {"crop": "Potato", "condition": "Early Blight", "status": "Diseased", "description": "Concentric dark target spot rings.", "prevention": "Crop rotation.", "treatment": "Copper fungicide."},
        "Potato___Late_blight": {"crop": "Potato", "condition": "Late Blight", "status": "Diseased", "description": "Dark water-soaked lesions.", "prevention": "Certified seeds.", "treatment": "Systemic fungicide."},
        "Potato___healthy": {"crop": "Potato", "condition": "Healthy Leaf", "status": "Healthy", "description": "Healthy foliage.", "prevention": "Proper drainage.", "treatment": "None needed."},
        "Tomato_Bacterial_spot": {"crop": "Tomato", "condition": "Bacterial Spot", "status": "Diseased", "description": "Dark greasy lesions.", "prevention": "Clean seeds.", "treatment": "Copper spray."},
        "Tomato_Early_blight": {"crop": "Tomato", "condition": "Early Blight", "status": "Diseased", "description": "Brown concentric rings.", "prevention": "Mulching.", "treatment": "Prune & copper spray."},
        "Tomato_Late_blight": {"crop": "Tomato", "condition": "Late Blight", "status": "Diseased", "description": "Dark irregular spots.", "prevention": "Good spacing.", "treatment": "Copper fungicide."},
        "Tomato_Leaf_Mold": {"crop": "Tomato", "condition": "Leaf Mold", "status": "Diseased", "description": "Yellow spots with velvety mold.", "prevention": "Ventilation.", "treatment": "Fungicide."},
        "Tomato_Septoria_leaf_spot": {"crop": "Tomato", "condition": "Septoria Leaf Spot", "status": "Diseased", "description": "Circular spots with dark margins.", "prevention": "Keep leaves dry.", "treatment": "Copper fungicide."},
        "Tomato_Spider_mites_Two_spotted_spider_mite": {"crop": "Tomato", "condition": "Spider Mites", "status": "Diseased", "description": "Yellow stippling & webbing.", "prevention": "Keep watered.", "treatment": "Neem oil/soap."},
        "Tomato__Target_Spot": {"crop": "Tomato", "condition": "Target Spot", "status": "Diseased", "description": "Brown spots with yellow halo.", "prevention": "Prune foliage.", "treatment": "Fungicide."},
        "Tomato__Tomato_YellowLeaf__Curl_Virus": {"crop": "Tomato", "condition": "Yellow Leaf Curl Virus", "status": "Diseased", "description": "Stunting & curling leaves.", "prevention": "Insect netting.", "treatment": "Whitefly control."},
        "Tomato__Tomato_mosaic_virus": {"crop": "Tomato", "condition": "Mosaic Virus", "status": "Diseased", "description": "Mottled green patterns.", "prevention": "Clean tools.", "treatment": "Remove infected plant."},
        "Tomato_healthy": {"crop": "Tomato", "condition": "Healthy Leaf", "status": "Healthy", "description": "Smooth deep green leaf.", "prevention": "Regular care.", "treatment": "None needed."}
    }

    def clean_class_name(raw_name: str) -> str:
        if raw_name in DISEASE_INFO:
            info = DISEASE_INFO[raw_name]
            return f"{info['crop']} - {info['condition']}"
        return raw_name.replace("___", " - ").replace("__", " - ").replace("_", " ")

    def extract_leaf_region(img: Image.Image) -> Image.Image:
        img_np = np.array(img.convert("RGB"))
        hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
        mask_green = cv2.inRange(hsv, np.array([20, 25, 25]), np.array([100, 255, 255]))
        mask_brown = cv2.inRange(hsv, np.array([5, 30, 30]), np.array([25, 255, 255]))
        mask = cv2.bitwise_or(mask_green, mask_brown)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            if cv2.contourArea(c) > (img_np.shape[0] * img_np.shape[1] * 0.04):
                x, y, w, h = cv2.boundingRect(c)
                pad = int(max(w, h) * 0.08)
                x1, y1 = max(0, x - pad), max(0, y - pad)
                x2, y2 = min(img_np.shape[1], x + w + pad), min(img_np.shape[0], y + h + pad)
                return Image.fromarray(img_np[y1:y2, x1:x2])
        return img

# Set page configuration
st.set_page_config(
    page_title="Plant Leaf Disease AI Classifier",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.3rem;
        color: #1b5e20;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #388e3c;
        text-align: center;
        margin-bottom: 25px;
    }
    .recommendation-card {
        background-color: #ffffff;
        color: #1f2937 !important;
        border: 1px solid #c8e6c9;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    }
    .recommendation-card p {
        color: #1f2937 !important;
    }
    .recommendation-card strong {
        color: #14532d !important;
    }
    .metric-card-healthy {
        background-color: #e8f5e9;
        color: #1f2937 !important;
        border-left: 6px solid #2e7d32;
        padding: 18px;
        border-radius: 8px;
        margin-bottom: 15px;
    }
    .metric-card-diseased {
        background-color: #ffebee;
        color: #1f2937 !important;
        border-left: 6px solid #c62828;
        padding: 18px;
        border-radius: 8px;
        margin-bottom: 15px;
    }
    .warning-card {
        background-color: #fffde7;
        color: #f57f17 !important;
        border-left: 5px solid #fbc02d;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 15px;
    }
    .status-healthy {
        color: #2e7d32;
        font-weight: bold;
        font-size: 1.1rem;
    }
    .status-diseased {
        color: #c62828;
        font-weight: bold;
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)

MODEL_PATH = "plant_disease_cnn.pth"
CLASS_NAMES_PATH = "class_names.json"
EVAL_RESULTS_PATH = "evaluation_results.json"
IMG_SIZE = 128

@st.cache_resource
def load_model_and_classes():
    """Cache and load PyTorch model and class metadata."""
    if not os.path.exists(CLASS_NAMES_PATH):
        return None, None, "Class metadata file missing. Please train the model first."
        
    with open(CLASS_NAMES_PATH, "r") as f:
        class_names = json.load(f)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_transfer_learning_model(num_classes=len(class_names), pretrained=False)
    
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.to(device)
        model.eval()
        return model, class_names, None
    else:
        return None, class_names, "Model weights missing. Please run train_model.py first."


def aspect_preserving_crop(image: Image.Image, size=128) -> Image.Image:
    """Center-crop image to square to preserve leaf aspect ratio without distortion."""
    w, h = image.size
    min_dim = min(w, h)
    left = (w - min_dim) / 2.0
    top = (h - min_dim) / 2.0
    right = (w + min_dim) / 2.0
    bottom = (h + min_dim) / 2.0
    cropped = image.crop((left, top, right, bottom))
    return cropped.resize((size, size), Image.Resampling.BILINEAR)


def get_base_transform(img_size=128):
    return T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def predict_image(image: Image.Image, model, class_names):
    """
    Predicts disease using OpenCV Leaf Region Extraction + Multi-Crop Test-Time Augmentation (TTA).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = get_base_transform(IMG_SIZE)
    
    # Extract leaf region using OpenCV
    leaf_crop = extract_leaf_region(image)
    rgb_img = leaf_crop.convert("RGB")
    
    # Multi-crop views
    crop1 = aspect_preserving_crop(rgb_img, IMG_SIZE)
    crop2 = rgb_img.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR)
    crop3 = crop1.transpose(Image.FLIP_LEFT_RIGHT)
    
    tensors = [
        transform(crop1).unsqueeze(0).to(device),
        transform(crop2).unsqueeze(0).to(device),
        transform(crop3).unsqueeze(0).to(device)
    ]
    
    logits_list = []
    with torch.no_grad():
        for t in tensors:
            out = model(t)
            logits_list.append(torch.softmax(out, dim=1))
            
    avg_probs = torch.mean(torch.stack(logits_list), dim=0)[0].cpu().numpy()
    top_indices = np.argsort(avg_probs)[::-1]
    
    results = []
    healthy_prob_sum = 0.0
    diseased_prob_sum = 0.0
    
    for idx in top_indices:
        raw_cls = class_names[idx]
        p = float(avg_probs[idx])
        info = DISEASE_INFO.get(raw_cls, {"status": "Diseased"})
        if info["status"] == "Healthy":
            healthy_prob_sum += p
        else:
            diseased_prob_sum += p
            
        results.append({
            'raw_class': raw_cls,
            'clean_name': clean_class_name(raw_cls),
            'probability': p
        })
        
    binary_health = {
        'is_healthy': healthy_prob_sum > diseased_prob_sum,
        'healthy_confidence': healthy_prob_sum * 100,
        'diseased_confidence': diseased_prob_sum * 100
    }
    
    return results, binary_health, leaf_crop


def main():
    st.markdown('<p class="main-header">🌿 Plant Leaf Health & Disease AI Diagnosis System</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Deep Learning CNN with OpenCV Automatic Foliage Segmentation for Real-World & Google Images</p>', unsafe_allow_html=True)
    
    model, class_names, err = load_model_and_classes()
    
    # Sidebar
    st.sidebar.image("https://img.icons8.com/color/96/000000/leaf.png", width=80)
    st.sidebar.title("Navigation & Controls")
    
    if err:
        st.sidebar.error(f"Status: {err}")
    else:
        st.sidebar.success(f"Model Status: Loaded & Ready ({len(class_names)} Classes)")
        
    st.sidebar.markdown("---")
    st.sidebar.subheader("Supported Crops")
    st.sidebar.markdown("- 🫑 **Pepper (Bell)** (2 conditions)")
    st.sidebar.markdown("- 🥔 **Potato** (3 conditions)")
    st.sidebar.markdown("- 🍅 **Tomato** (10 conditions)")
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["🔬 Leaf Diagnosis (Healthy vs. Diseased)", "📊 Model Evaluation & Analytics", "ℹ️ How Google Image Generalization Works"])
    
    with tab1:
        col1, col2 = st.columns([1, 1.2])
        
        with col1:
            st.subheader("Upload Plant Leaf Image")
            uploaded_file = st.file_uploader(
                "Choose a leaf image from your computer or downloaded from Google", type=["jpg", "jpeg", "png"]
            )
            
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption="Original Uploaded Image", use_container_width=True)
            else:
                st.info("💡 Upload any Pepper, Potato, or Tomato leaf photo to determine if it is **Healthy** or **Diseased**.")
                
        with col2:
            st.subheader("AI Health Inspection Results")
            
            if uploaded_file is not None:
                if model is None:
                    st.error("Model is not loaded. Please ensure `plant_disease_cnn.pth` exists.")
                else:
                    with st.spinner("Extracting leaf surface & evaluating foliar health..."):
                        predictions, binary_health, leaf_crop = predict_image(image, model, class_names)
                        top_pred = predictions[0]
                        raw_cls = top_pred['raw_class']
                        top_prob = top_pred['probability'] * 100
                        
                        # Display Segmented Leaf Preview
                        with st.expander("🔍 OpenCV Automated Leaf Extraction Preview", expanded=False):
                            st.image(leaf_crop, caption="Segmented Leaf Surface (Soil & Background Stripped)", width=250)
                        
                        info = DISEASE_INFO.get(raw_cls, {
                            "crop": "Unknown",
                            "condition": top_pred['clean_name'],
                            "status": "Unknown",
                            "description": "N/A",
                            "prevention": "N/A",
                            "treatment": "N/A"
                        })
                        
                        # Top-level HEALTHY vs DISEASED Banner
                        if binary_health['is_healthy']:
                            card_style = "metric-card-healthy"
                            health_badge = "🟢 HEALTHY LEAF"
                            health_conf = binary_health['healthy_confidence']
                        else:
                            card_style = "metric-card-diseased"
                            health_badge = "🔴 DISEASED LEAF"
                            health_conf = binary_health['diseased_confidence']
                            
                        st.markdown(f"""
                        <div class="{card_style}">
                            <h2 style="margin:0px; font-weight:800;">{health_badge} ({health_conf:.1f}% Confidence)</h2>
                            <p style="margin-top:5px; font-size:1.1rem;">
                                <strong>Specific Condition:</strong> {info['crop']} - {info['condition']} (Top Class Match: {top_prob:.1f}%)
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Out-of-distribution Warning for Non-Leaf / Low Confidence
                        if top_prob < 45.0:
                            st.markdown("""
                            <div class="warning-card">
                                <strong>⚠️ Low Confidence Prediction (< 45%):</strong><br/>
                                This photo may contain non-foliar objects or severe background noise.<br/>
                                <em>Tip: Crop closer to a single leaf surface for optimal accuracy.</em>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Probability Distribution Chart
                        st.write("##### Top 5 Condition Predictions")
                        df_top5 = pd.DataFrame(predictions[:5])
                        df_top5['percentage'] = df_top5['probability'] * 100
                        
                        fig, ax = plt.subplots(figsize=(7, 3))
                        sns.barplot(
                            x='percentage', y='clean_name', data=df_top5,
                            palette="Greens_r" if binary_health['is_healthy'] else "Reds_r", ax=ax
                        )
                        ax.set_xlabel("Probability (%)")
                        ax.set_ylabel("")
                        ax.set_xlim(0, 100)
                        for p in ax.patches:
                            w_val = p.get_width()
                            ax.annotate(f'{w_val:.1f}%', (w_val + 1, p.get_y() + p.get_height()/2),
                                        ha='left', va='center', fontsize=9)
                        sns.despine()
                        st.pyplot(fig)
                        
                        # Actionable Agricultural Advice Card
                        st.markdown("##### 📋 Agricultural Care & Management Advice")
                        st.markdown(f"""
                        <div class="recommendation-card">
                            <p><strong>Overview:</strong> {info['description']}</p>
                            <p><strong>🛡️ Prevention:</strong> {info['prevention']}</p>
                            <p><strong>💊 Recommended Action:</strong> {info['treatment']}</p>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.write("Upload a leaf image on the left panel to begin health inspection.")

    with tab2:
        st.subheader("Model Evaluation & Training Metrics")
        
        if os.path.exists(EVAL_RESULTS_PATH):
            with open(EVAL_RESULTS_PATH, "r") as f:
                eval_data = json.load(f)
                
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Overall Test Accuracy", f"{eval_data['test_accuracy']*100:.2f}%")
            m2.metric("Weighted Precision", f"{eval_data['precision_weighted']*100:.2f}%")
            m3.metric("Weighted Recall", f"{eval_data['recall_weighted']*100:.2f}%")
            m4.metric("Weighted F1 Score", f"{eval_data['f1_score_weighted']*100:.2f}%")
            
            st.markdown("---")
            st.write("##### Loss & Accuracy Curves Across Training Epochs")
            
            hist = eval_data['history']
            epochs = list(range(1, len(hist['train_loss']) + 1))
            
            fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            ax1.plot(epochs, hist['train_loss'], 'o-', label='Train Loss', color='#1e88e5')
            ax1.plot(epochs, hist['val_loss'], 's-', label='Validation Loss', color='#e53935')
            ax1.set_title("Cross-Entropy Loss Curve")
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Loss")
            ax1.legend()
            ax1.grid(True, linestyle='--', alpha=0.5)
            
            ax2.plot(epochs, [a * 100 for a in hist['train_acc']], 'o-', label='Train Accuracy', color='#43a047')
            ax2.plot(epochs, [a * 100 for a in hist['val_acc']], 's-', label='Validation Accuracy', color='#fb8c00')
            ax2.set_title("Classification Accuracy Curve (%)")
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("Accuracy (%)")
            ax2.legend()
            ax2.grid(True, linestyle='--', alpha=0.5)
            
            st.pyplot(fig2)

    with tab3:
        st.subheader("How Real-World & Google Image Generalization Works")
        st.markdown("""
        ### Why PlantVillage Models Overfit & How We Solved It
        - **The PlantVillage Overfitting Problem**: Standard models trained on PlantVillage memorize the neutral studio background and studio lighting. When given Google images containing soil, garden backgrounds, or hands, standard models fail.
        - **Solutions Implemented in This System**:
          1. **Automated OpenCV HSV Foliage Segmentation**: Automatically isolates green and brown leaf regions, cropping out background soil, sky, or human hands.
          2. **Test-Time Augmentation (TTA)**: Averages predictions across 3 spatial crops to ensure multi-angle accuracy.
          3. **Binary Health Classification**: Aggregates 15 sub-classes into high-level **HEALTHY vs. DISEASED** status.
        """)

if __name__ == "__main__":
    main()
