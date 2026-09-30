"""
Plant Disease Treatment & Care Recommendations Dictionary
Enhanced with OpenCV Leaf Region Extractor for Google/Real-World Photos.
"""

from PIL import Image

try:
    import cv2
    import numpy as np
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

DISEASE_INFO = {
    "Pepper__bell___Bacterial_spot": {
        "crop": "Pepper (Bell)",
        "condition": "Bacterial Spot",
        "status": "Diseased",
        "description": "Small spots turning brown and necrotic.",
        "prevention": "Use pathogen-free seeds, practice crop rotation, and avoid overhead watering.",
        "treatment": "Apply copper-based fungicides/bactericides early. Remove infected debris."
    },
    "Pepper__bell___healthy": {
        "crop": "Pepper (Bell)",
        "condition": "Healthy Leaf",
        "status": "Healthy",
        "description": "Leaf shows optimal health, vibrant green coloration, and no signs of bacterial or fungal infection.",
        "prevention": "Maintain proper soil moisture, balanced fertilization, and good ventilation.",
        "treatment": "No treatment required. Continue current care practices."
    },
    "Potato___Early_blight": {
        "crop": "Potato",
        "condition": "Early Blight",
        "status": "Diseased",
        "description": "Features concentric dark target-spot rings on lower leaves.",
        "prevention": "Plant resistant cultivars, rotate crops, and ensure proper soil nitrogen.",
        "treatment": "Apply protective fungicides containing chlorothalonil or copper."
    },
    "Potato___Late_blight": {
        "crop": "Potato",
        "condition": "Late Blight",
        "status": "Diseased",
        "description": "Produces dark water-soaked lesions with white mold underneath.",
        "prevention": "Use certified seed tubers and avoid foliage wetness.",
        "treatment": "Apply systemic copper fungicides immediately. Destroy heavily infected plants."
    },
    "Potato___healthy": {
        "crop": "Potato",
        "condition": "Healthy Leaf",
        "status": "Healthy",
        "description": "Foliage is green, robust, and free from fungal spots or lesions.",
        "prevention": "Ensure good drainage, hilling around tubers, and routine scouting.",
        "treatment": "No treatment needed. Maintain optimal care."
    },
    "Tomato_Bacterial_spot": {
        "crop": "Tomato",
        "condition": "Bacterial Spot",
        "status": "Diseased",
        "description": "Dark greasy lesions on leaves causing defoliation.",
        "prevention": "Use disease-free seeds and eliminate weed hosts.",
        "treatment": "Spray copper-based formulations combined with mancozeb."
    },
    "Tomato_Early_blight": {
        "crop": "Tomato",
        "condition": "Early Blight",
        "status": "Diseased",
        "description": "Brown concentric rings causing premature leaf drop.",
        "prevention": "Mulch plant bases, stake off ground, and rotate crops.",
        "treatment": "Prune lower infected leaves. Treat with copper fungicides."
    },
    "Tomato_Late_blight": {
        "crop": "Tomato",
        "condition": "Late Blight",
        "status": "Diseased",
        "description": "Dark irregular water-soaked spots turning papery.",
        "prevention": "Ensure high sunlight, ample spacing, and drip irrigation.",
        "treatment": "Apply copper fungicides immediately. Remove infected plants."
    },
    "Tomato_Leaf_Mold": {
        "crop": "Tomato",
        "condition": "Leaf Mold",
        "status": "Diseased",
        "description": "Pale yellow spots with olive-green velvet mold underneath.",
        "prevention": "Improve ventilation, keep humidity below 85%, increase row spacing.",
        "treatment": "Apply fungicides containing chlorothalonil or copper."
    },
    "Tomato_Septoria_leaf_spot": {
        "crop": "Tomato",
        "condition": "Septoria Leaf Spot",
        "status": "Diseased",
        "description": "Circular spots with dark margins and gray centers.",
        "prevention": "Keep foliage dry, control solanaceous weeds, mulch base.",
        "treatment": "Remove lower infected leaves. Apply copper or sulfur fungicides."
    },
    "Tomato_Spider_mites_Two_spotted_spider_mite": {
        "crop": "Tomato",
        "condition": "Spider Mites",
        "status": "Diseased",
        "description": "Yellow stippling and fine silk webbing.",
        "prevention": "Keep plants well-watered and avoid excess nitrogen.",
        "treatment": "Spray under-sides of leaves with insecticidal soap, neem oil, or miticides."
    },
    "Tomato__Target_Spot": {
        "crop": "Tomato",
        "condition": "Target Spot",
        "status": "Diseased",
        "description": "Pinpoint brown spots expanding with yellow halos.",
        "prevention": "Avoid excessive shade and prune dense foliage for airflow.",
        "treatment": "Apply recommended broad-spectrum fungicides (azoxystrobin, copper)."
    },
    "Tomato__Tomato_YellowLeaf__Curl_Virus": {
        "crop": "Tomato",
        "condition": "Yellow Leaf Curl Virus",
        "status": "Diseased",
        "description": "Causes severe stunting, leaf curling, and yellow margins.",
        "prevention": "Use reflective mulches, insect netting, and resistant hybrids.",
        "treatment": "Control whitefly vectors using neem oil or yellow sticky traps."
    },
    "Tomato__Tomato_mosaic_virus": {
        "crop": "Tomato",
        "condition": "Mosaic Virus",
        "status": "Diseased",
        "description": "Mottled light/dark green mosaic patterns and leaf distortion.",
        "prevention": "Wash hands with soap, sanitize tools, avoid tobacco near plants.",
        "treatment": "Destroy infected plants immediately to halt spread."
    },
    "Tomato_healthy": {
        "crop": "Tomato",
        "condition": "Healthy Leaf",
        "status": "Healthy",
        "description": "Deep green color, smooth surface, and no signs of disease or pest activity.",
        "prevention": "Maintain regular watering, balanced soil pH, and routine inspection.",
        "treatment": "No intervention needed. Continue regular care."
    }
}

def clean_class_name(raw_name: str) -> str:
    """Format raw folder name into a clean title."""
    if raw_name in DISEASE_INFO:
        info = DISEASE_INFO[raw_name]
        return f"{info['crop']} - {info['condition']}"
    return raw_name.replace("___", " - ").replace("__", " - ").replace("_", " ")


def extract_leaf_region(image: Image.Image) -> Image.Image:
    """
    Uses OpenCV HSV color segmentation to isolate the main leaf region from real-world / Google images.
    Falls back gracefully if OpenCV is not installed.
    """
    if not OPENCV_AVAILABLE:
        return image
        
    try:
        img_rgb = image.convert("RGB")
        img_np = np.array(img_rgb)
        hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
        
        # Green foliage mask
        lower_green = np.array([20, 25, 25])
        upper_green = np.array([100, 255, 255])
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        
        # Yellow / Brown lesion mask
        lower_brown = np.array([5, 30, 30])
        upper_brown = np.array([25, 255, 255])
        mask_brown = cv2.inRange(hsv, lower_brown, upper_brown)
        
        mask = cv2.bitwise_or(mask_green, mask_brown)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            if area > (img_np.shape[0] * img_np.shape[1] * 0.04):
                x, y, w, h = cv2.boundingRect(largest_contour)
                pad = int(max(w, h) * 0.08)
                x1, y1 = max(0, x - pad), max(0, y - pad)
                x2, y2 = min(img_np.shape[1], x + w + pad), min(img_np.shape[0], y + h + pad)
                cropped_np = img_np[y1:y2, x1:x2]
                return Image.fromarray(cropped_np)
    except Exception:
        pass
        
    return image
