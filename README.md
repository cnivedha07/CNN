# 🌿 Plant Disease Classification using Deep Learning

## 📌 Project Overview

This project presents an AI-powered plant disease classification system that uses deep learning and transfer learning to identify diseases from plant leaf images.

The system uses the PlantVillage dataset and a MobileNetV2-based convolutional neural network to classify images into 15 plant health/disease categories.

A Streamlit web application is also provided, allowing users to upload a plant leaf image and receive a predicted disease along with prediction confidence and agricultural recommendations.

---

## 🎯 Objectives

- Automatically identify plant diseases from leaf images.
- Classify plant leaves into 15 disease and healthy categories.
- Apply image preprocessing and data augmentation.
- Use transfer learning with MobileNetV2.
- Evaluate the model using accuracy, precision, recall, F1-score, and confusion matrix.
- Develop an interactive web application for real-time prediction.
- Provide disease-specific agricultural advice and recommendations.

---

## 📊 Dataset

The project uses the **PlantVillage dataset**.

The dataset contains approximately **20,600 images** belonging to 15 classes covering Pepper, Potato, and Tomato plants.

### Supported Classes

1. Pepper Bell Bacterial Spot
2. Pepper Bell Healthy
3. Potato Early Blight
4. Potato Late Blight
5. Potato Healthy
6. Tomato Bacterial Spot
7. Tomato Early Blight
8. Tomato Late Blight
9. Tomato Leaf Mold
10. Tomato Septoria Leaf Spot
11. Tomato Spider Mites
12. Tomato Target Spot
13. Tomato Yellow Leaf Curl Virus
14. Tomato Mosaic Virus
15. Tomato Healthy

---

## 🔄 Data Preprocessing

The following preprocessing techniques were applied:

- Image resizing to `128 × 128` pixels
- RGB conversion
- Training/validation/test split
- Random horizontal flipping
- Random vertical flipping
- Random rotation
- Color jittering
- ImageNet normalization

The dataset was divided into:

- **80% Training**
- **10% Validation**
- **10% Testing**

---

## 🧠 Model Architecture

### MobileNetV2 Transfer Learning

The project uses **MobileNetV2** as the deep learning architecture.

MobileNetV2 provides an efficient convolutional neural network architecture suitable for image classification while maintaining relatively low computational requirements.

The model was adapted for the 15 plant disease classes by replacing the final classification layer.

### Model Configuration

| Parameter | Value |
|---|---|
| Architecture | MobileNetV2 |
| Input Size | 128 × 128 |
| Number of Classes | 15 |
| Loss Function | CrossEntropyLoss |
| Optimizer | AdamW |
| Learning Rate | 0.001 |
| Weight Decay | 0.0001 |
| Scheduler | ReduceLROnPlateau |
| Batch Size | 64 |
| Epochs | 4 |

---

## 📈 Model Performance

The model achieved the following results on the unseen test dataset:

| Metric | Result |
|---|---:|
| Test Accuracy | **98.59%** |
| Weighted Precision | **98.61%** |
| Weighted Recall | **98.59%** |
| Weighted F1-Score | **98.59%** |

### Training Performance

| Epoch | Training Accuracy | Validation Accuracy |
|---:|---:|---:|
| 1 | 89.86% | 95.88% |
| 2 | 96.15% | 97.04% |
| 3 | 96.71% | 97.97% |
| 4 | 97.69% | 98.93% |

The confusion matrix and classification report are included in the project for detailed performance analysis.

---

## 🌐 Web Application

The project includes an interactive **Streamlit web application**.

The application allows users to:

1. Upload a plant leaf image.
2. Process the image using the trained model.
3. Predict the plant disease/health category.
4. Display the prediction confidence.
5. Display the probability distribution of the top five predictions.
6. Provide disease-related agricultural recommendations.

---

## 📁 Project Structure

```text
CNN/
│
├── PlantVillage/
│
├── src/
│   ├── __init__.py
│   ├── dataset.py
│   ├── model.py
│   └── utils.py
│
├── Plant_Disease_Classification_CNN.ipynb
├── train_model.py
├── analyze_model.py
├── app.py
├── gradio_app.py
├── plant_disease_cnn.pth
├── class_names.json
├── evaluation_results.json
├── requirements.txt
└── README.md