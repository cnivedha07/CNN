import os
import json
import shutil

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report
from PIL import Image

from src.dataset import prepare_dataloaders
from src.model import get_transfer_learning_model
from src.utils import clean_class_name


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = r"c:\Users\cnive\Ai training\CNN\PlantVillage"
MODEL_PATH = r"c:\Users\cnive\Ai training\CNN\plant_disease_cnn.pth"
CLASS_NAMES_PATH = r"c:\Users\cnive\Ai training\CNN\class_names.json"

IMG_SIZE = 128
BATCH_SIZE = 64

OUTPUT_DIR = r"c:\Users\cnive\Ai training\CNN\model_analysis"
MISCLASSIFIED_DIR = os.path.join(OUTPUT_DIR, "misclassified_images")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MISCLASSIFIED_DIR, exist_ok=True)


# ============================================================
# DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 70)
print("        PLANT DISEASE MODEL - ERROR ANALYSIS")
print("=" * 70)
print(f"[+] Device: {device}")


# ============================================================
# LOAD DATASET
# ============================================================

print("\n[+] Recreating the original test split...")

train_loader, val_loader, test_loader, split_info = prepare_dataloaders(
    DATA_DIR,
    img_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

class_names = split_info["class_names"]
num_classes = split_info["num_classes"]

print(f"[+] Test images: {split_info['test_size']}")
print(f"[+] Number of classes: {num_classes}")


# ============================================================
# LOAD MODEL
# ============================================================

print("\n[+] Loading MobileNetV2 model...")

model = get_transfer_learning_model(
    num_classes=num_classes,
    pretrained=False
)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=device)
)

model = model.to(device)
model.eval()

print("[+] Model weights loaded successfully.")


# ============================================================
# PREDICTIONS
# ============================================================

print("\n[+] Running predictions on the test dataset...")

all_predictions = []
all_labels = []
all_confidences = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(images)

        probabilities = torch.softmax(outputs, dim=1)

        confidences, predictions = torch.max(probabilities, dim=1)

        all_predictions.extend(
            predictions.cpu().numpy().tolist()
        )

        all_labels.extend(
            labels.numpy().tolist()
        )

        all_confidences.extend(
            confidences.cpu().numpy().tolist()
        )


y_true = np.array(all_labels)
y_pred = np.array(all_predictions)
confidences = np.array(all_confidences)


# ============================================================
# BASIC RESULTS
# ============================================================

correct = y_true == y_pred

accuracy = np.mean(correct)

print("\n" + "=" * 70)
print("OVERALL RESULTS")
print("=" * 70)

print(f"Test Accuracy : {accuracy * 100:.2f}%")
print(f"Correct       : {np.sum(correct)}")
print(f"Incorrect     : {np.sum(~correct)}")


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

clean_labels = [
    clean_class_name(name)
    for name in class_names
]

report = classification_report(
    y_true,
    y_pred,
    target_names=clean_labels,
    digits=4
)

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(report)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=list(range(num_classes))
)

print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print(cm)


# ============================================================
# SAVE CONFUSION MATRIX IMAGE
# ============================================================

fig_size = max(12, num_classes * 0.8)

plt.figure(figsize=(fig_size, fig_size))

plt.imshow(cm)

plt.title("Plant Disease Classification - Confusion Matrix")

plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")

plt.xticks(
    range(num_classes),
    clean_labels,
    rotation=90
)

plt.yticks(
    range(num_classes),
    clean_labels
)

# Write numbers inside cells
for i in range(num_classes):
    for j in range(num_classes):

        plt.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center"
        )

plt.colorbar()

plt.tight_layout()

cm_path = os.path.join(
    OUTPUT_DIR,
    "confusion_matrix.png"
)

plt.savefig(cm_path, dpi=200)

plt.close()

print(f"\n[+] Confusion matrix saved to:")
print(f"    {cm_path}")


# ============================================================
# FIND MOST CONFUSED CLASS PAIRS
# ============================================================

confusions = []

for actual in range(num_classes):

    for predicted in range(num_classes):

        if actual == predicted:
            continue

        count = cm[actual, predicted]

        if count > 0:

            confusions.append({
                "actual": clean_labels[actual],
                "predicted": clean_labels[predicted],
                "count": int(count)
            })


confusions.sort(
    key=lambda x: x["count"],
    reverse=True
)

print("\n" + "=" * 70)
print("MOST COMMON CONFUSIONS")
print("=" * 70)

for item in confusions[:15]:

    print(
        f"{item['actual']}"
        f"  -->  "
        f"{item['predicted']}"
        f" : {item['count']} images"
    )


# ============================================================
# SAVE MISCLASSIFIED IMAGES
# ============================================================

print("\n[+] Locating misclassified images...")

# Recreate the exact test dataset paths.
test_dataset = test_loader.dataset

misclassified_records = []

for index in range(len(test_dataset)):

    actual = y_true[index]
    predicted = y_pred[index]

    if actual == predicted:
        continue

    image_path = test_dataset.image_paths[index]

    confidence = float(confidences[index])

    actual_name = clean_labels[actual]
    predicted_name = clean_labels[predicted]

    # Create safe folder names
    actual_folder = actual_name.replace("/", "_").replace("\\", "_")
    predicted_folder = predicted_name.replace("/", "_").replace("\\", "_")

    output_folder = os.path.join(
        MISCLASSIFIED_DIR,
        f"Actual_{actual_folder}__Predicted_{predicted_folder}"
    )

    os.makedirs(output_folder, exist_ok=True)

    filename = os.path.basename(image_path)

    destination = os.path.join(
        output_folder,
        filename
    )

    shutil.copy2(
        image_path,
        destination
    )

    misclassified_records.append({
        "image": image_path,
        "actual": actual_name,
        "predicted": predicted_name,
        "confidence": confidence
    })


# ============================================================
# SAVE JSON ANALYSIS
# ============================================================

analysis = {
    "test_accuracy": float(accuracy),
    "test_images": int(len(y_true)),
    "correct_predictions": int(np.sum(correct)),
    "incorrect_predictions": int(np.sum(~correct)),
    "most_common_confusions": confusions[:15],
    "misclassified_images": misclassified_records
}

json_path = os.path.join(
    OUTPUT_DIR,
    "evaluation_analysis.json"
)

with open(json_path, "w") as f:

    json.dump(
        analysis,
        f,
        indent=4
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)

print(f"\n[+] Confusion matrix:")
print(f"    {cm_path}")

print(f"\n[+] Misclassified images:")
print(f"    {MISCLASSIFIED_DIR}")

print(f"\n[+] Detailed JSON:")
print(f"    {json_path}")

print("\n[+] Done.")