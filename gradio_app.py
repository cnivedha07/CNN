"""
Plant Disease Classification & Health Diagnosis - Gradio Web Application
Powered by Custom Deep Convolutional Neural Network (PlantDiseaseCNN),
OpenCV Automatic Leaf Segmentation, Test-Time Augmentation (TTA),
and Binary Health Diagnosis for Any Image (Dataset or Google Photos).
"""

import os
import sys
import json

project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import numpy as np
import gradio as gr
from PIL import Image
import torchvision.transforms as T

from src.model import PlantDiseaseCNN
from src.utils import DISEASE_INFO, clean_class_name, extract_leaf_region

MODEL_PATH = "plant_disease_cnn.pth"
CLASS_NAMES_PATH = "class_names.json"
IMG_SIZE = 128

def load_model():
    if not os.path.exists(CLASS_NAMES_PATH) or not os.path.exists(MODEL_PATH):
        return None, None
        
    with open(CLASS_NAMES_PATH, "r") as f:
        class_names = json.load(f)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PlantDiseaseCNN(num_classes=len(class_names))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model, class_names

model, class_names = load_model()

def aspect_preserving_crop(image: Image.Image, size=128) -> Image.Image:
    w, h = image.size
    min_dim = min(w, h)
    left = (w - min_dim) / 2.0
    top = (h - min_dim) / 2.0
    right = (w + min_dim) / 2.0
    bottom = (h + min_dim) / 2.0
    cropped = image.crop((left, top, right, bottom))
    return cropped.resize((size, size), Image.Resampling.BILINEAR)

transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def diagnose_leaf(image):
    if image is None:
        return "Please upload a valid leaf image.", {}, None
        
    if model is None:
        return "Model weights not found. Please train the model first.", {}, None
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    leaf_crop = extract_leaf_region(image)
    rgb_img = leaf_crop.convert("RGB")
    
    crop1 = aspect_preserving_crop(rgb_img, IMG_SIZE)
    crop2 = rgb_img.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR)
    crop3 = crop1.transpose(Image.FLIP_LEFT_RIGHT)
    
    tensors = [
        transform(crop1).unsqueeze(0).to(device),
        transform(crop2).unsqueeze(0).to(device),
        transform(crop3).unsqueeze(0).to(device)
    ]
    
    logits = []
    with torch.no_grad():
        for t in tensors:
            outs = model(t)
            logits.append(torch.softmax(outs, dim=1))
            
    probabilities = torch.mean(torch.stack(logits), dim=0)[0].cpu().numpy()
    
    prob_dict = {}
    healthy_sum = 0.0
    diseased_sum = 0.0
    
    for idx, prob in enumerate(probabilities):
        name = clean_class_name(class_names[idx])
        prob_dict[name] = float(prob)
        raw_cls = class_names[idx]
        info = DISEASE_INFO.get(raw_cls, {"status": "Diseased"})
        if info["status"] == "Healthy":
            healthy_sum += prob
        else:
            diseased_sum += prob
            
    top_idx = int(probabilities.argmax())
    top_raw = class_names[top_idx]
    top_prob = float(probabilities[top_idx]) * 100
    
    is_healthy = healthy_sum > diseased_sum
    status_icon = "🟢 HEALTHY LEAF" if is_healthy else "🔴 DISEASED LEAF"
    health_conf = (healthy_sum if is_healthy else diseased_sum) * 100
    
    info = DISEASE_INFO.get(top_raw, {
        "crop": "Unknown",
        "condition": clean_class_name(top_raw),
        "status": "Unknown",
        "description": "N/A",
        "prevention": "N/A",
        "treatment": "N/A"
    })
    
    summary = f"## {status_icon} ({health_conf:.1f}% Confidence)\n\n"
    summary += f"- **Specific Condition Match:** {info['crop']} - {info['condition']} ({top_prob:.1f}% match)\n"
    
    if top_prob < 45.0:
        summary += f"\n> ⚠️ **Low Confidence Warning (< 45%):** Photo may contain background clutter. Crop closer to the leaf surface for best results.\n"
        
    summary += f"\n### 📋 Actionable Agricultural Care Advice\n"
    summary += f"- **Overview:** {info['description']}\n"
    summary += f"- **Prevention:** {info['prevention']}\n"
    summary += f"- **Treatment:** {info['treatment']}\n"
    
    return summary, prob_dict, leaf_crop

demo = gr.Interface(
    fn=diagnose_leaf,
    inputs=gr.Image(type="pil", label="Upload Plant Leaf Image (Local or Google)"),
    outputs=[
        gr.Markdown(label="AI Health Diagnosis & Treatment Plan"),
        gr.Label(num_top_classes=5, label="Top Candidate Probabilities"),
        gr.Image(label="OpenCV Segmented Leaf Area")
    ],
    title="🌿 Plant Leaf Health AI Classifier & Diagnosis System",
    description="Upload any leaf photo (Pepper, Potato, Tomato) to determine whether it is Healthy or Diseased.",
    examples=[],
    theme="soft"
)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", port=7860, share=False)
