"""
Dataset indexing, transformation pipelines, and PyTorch DataLoaders
for the PlantVillage dataset.
"""

import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from sklearn.model_selection import train_test_split

VALID_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')

class PlantVillageDataset(Dataset):
    """Custom PyTorch Dataset for PlantVillage Leaf Images."""
    
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


def get_transforms(img_size=128):
    """
    Returns data transformation pipelines for training, validation, and testing.
    """
    train_transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.3),
        T.RandomRotation(degrees=15),
        T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    eval_transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, eval_transform


def scan_plant_village(data_dir):
    class_names = []
    image_paths = []
    labels = []
    
    entries = sorted(os.listdir(data_dir))
    for entry in entries:
        full_path = os.path.join(data_dir, entry)
        if not os.path.isdir(full_path) or entry == "PlantVillage":
            continue
            
        class_name = entry
        class_names.append(class_name)
        class_idx = len(class_names) - 1
        
        for file in os.listdir(full_path):
            if file.lower().endswith(VALID_EXTENSIONS):
                img_path = os.path.join(full_path, file)
                image_paths.append(img_path)
                labels.append(class_idx)
                
    return class_names, image_paths, labels


def prepare_dataloaders(data_dir, img_size=128, batch_size=64, num_workers=0, seed=42):
    class_names, image_paths, labels = scan_plant_village(data_dir)
    
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        image_paths, labels, test_size=0.20, random_state=seed, stratify=labels
    )
    
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=0.50, random_state=seed, stratify=temp_labels
    )
    
    train_transform, eval_transform = get_transforms(img_size=img_size)
    
    train_dataset = PlantVillageDataset(train_paths, train_labels, transform=train_transform)
    val_dataset = PlantVillageDataset(val_paths, val_labels, transform=eval_transform)
    test_dataset = PlantVillageDataset(test_paths, test_labels, transform=eval_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    split_info = {
        'train_size': len(train_paths),
        'val_size': len(val_paths),
        'test_size': len(test_paths),
        'total_images': len(image_paths),
        'num_classes': len(class_names),
        'class_names': class_names
    }
    
    return train_loader, val_loader, test_loader, split_info
