"""
Standalone Training and Evaluation Script for Plant Disease Classification CNN.
Trains custom PlantDiseaseCNN model on PlantVillage dataset,
evaluates test accuracy, saves weights & metadata.
"""

import os
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support

from src.dataset import prepare_dataloaders
from src.model import PlantDiseaseCNN
from src.utils import clean_class_name

DATA_DIR = r"c:\Users\cnive\Ai training\CNN\PlantVillage"
MODEL_SAVE_PATH = r"c:\Users\cnive\Ai training\CNN\plant_disease_cnn.pth"
CLASS_NAMES_PATH = r"c:\Users\cnive\Ai training\CNN\class_names.json"
EVAL_RESULTS_PATH = r"c:\Users\cnive\Ai training\CNN\evaluation_results.json"

IMG_SIZE = 128
BATCH_SIZE = 64
EPOCHS = 6
LEARNING_RATE = 0.001

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels.data).item()
        total += labels.size(0)
        
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data).item()
            total += labels.size(0)
            
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
            
    eval_loss = running_loss / total
    eval_acc = correct / total
    return eval_loss, eval_acc, all_preds, all_labels


def main():
    print("=" * 70, flush=True)
    print("      PLANT DISEASE CLASSIFICATION - CUSTOM CNN TRAINING PIPELINE", flush=True)
    print("=" * 70, flush=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[+] Active Computing Device: {device}", flush=True)
    
    # 1. Prepare Data Loaders
    print("[+] Scanning PlantVillage dataset and preparing DataLoaders...", flush=True)
    train_loader, val_loader, test_loader, split_info = prepare_dataloaders(
        DATA_DIR, img_size=IMG_SIZE, batch_size=BATCH_SIZE
    )
    
    class_names = split_info['class_names']
    num_classes = split_info['num_classes']
    print(f"[+] Total Images Found : {split_info['total_images']}", flush=True)
    print(f"[+] Training Images     : {split_info['train_size']}", flush=True)
    print(f"[+] Validation Images   : {split_info['val_size']}", flush=True)
    print(f"[+] Testing Images      : {split_info['test_size']}", flush=True)
    print(f"[+] Number of Classes   : {num_classes}", flush=True)
    
    # Save class names JSON
    with open(CLASS_NAMES_PATH, "w") as f:
        json.dump(class_names, f, indent=4)
    print(f"[+] Class names metadata saved to: {CLASS_NAMES_PATH}", flush=True)
    
    # 2. Instantiate Custom CNN Model
    model = PlantDiseaseCNN(num_classes=num_classes, dropout_rate=0.4).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=1)
    
    print("\n[+] Custom Model Architecture initialized (PlantDiseaseCNN)", flush=True)
    print(model, flush=True)
    
    # 3. Training Loop
    print("\n" + "-" * 70, flush=True)
    print(f" Starting Training for {EPOCHS} Epochs...", flush=True)
    print("-" * 70, flush=True)
    
    best_val_acc = 0.0
    start_time = time.time()
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()
        
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate_model(model, val_loader, criterion, device)
        
        scheduler.step(val_acc)
        epoch_time = time.time() - epoch_start
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch [{epoch:02d}/{EPOCHS:02d}] ({epoch_time:.1f}s) - "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%", flush=True)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"    --> Best model checkpoint saved to: {MODEL_SAVE_PATH} (Val Acc: {val_acc*100:.2f}%)", flush=True)
            
    total_train_time = time.time() - start_time
    print(f"\n[+] Training completed in {total_train_time / 60:.2f} minutes.", flush=True)
    
    # 4. Final Evaluation on Test Dataset
    print("\n" + "=" * 70, flush=True)
    print("      FINAL EVALUATION ON UNSEEN TEST DATASET", flush=True)
    print("=" * 70, flush=True)
    
    # Load best saved weights
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    test_loss, test_acc, y_pred, y_true = evaluate_model(model, test_loader, criterion, device)
    
    print(f"\n[>>>] Overall Test Loss     : {test_loss:.4f}", flush=True)
    print(f"[>>>] Overall Test Accuracy : {test_acc * 100:.2f}%\n", flush=True)
    
    clean_labels = [clean_class_name(c) for c in class_names]
    report_str = classification_report(y_true, y_pred, target_names=clean_labels, digits=4)
    print("Classification Report:", flush=True)
    print(report_str, flush=True)
    
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    
    eval_summary = {
        'test_loss': float(test_loss),
        'test_accuracy': float(test_acc),
        'precision_weighted': float(precision),
        'recall_weighted': float(recall),
        'f1_score_weighted': float(f1),
        'epochs_trained': EPOCHS,
        'training_time_seconds': float(total_train_time),
        'history': history
    }
    
    with open(EVAL_RESULTS_PATH, "w") as f:
        json.dump(eval_summary, f, indent=4)
    print(f"[+] Full Evaluation Summary saved to: {EVAL_RESULTS_PATH}", flush=True)

if __name__ == "__main__":
    main()
