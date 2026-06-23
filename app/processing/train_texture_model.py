"""
Texture Severity Classifier — Transfer Learning
Trains MobileNetV2 on labeled wrinkles/texture images.
Run this ONCE to produce texture_model.pth.
"""

import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image

# ── Configuration ────────────────────────────────────────────────────────────
DATASET_DIR  = 'dataset/texture'
MODEL_OUTPUT = 'app/processing/texture_model.pth'
NUM_CLASSES  = 3  # e.g., 0=Smooth, 1=Fine Lines, 2=Deep Wrinkles
EPOCHS       = 15
BATCH_SIZE   = 16
LR           = 0.001
IMG_SIZE     = 224
DEVICE       = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Training Texture Model on: {DEVICE}")

class TextureDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples   = []
        self.transform = transform

        for label in range(NUM_CLASSES):
            folder = os.path.join(root_dir, str(label))
            if not os.path.exists(folder):
                continue
            for fname in os.listdir(folder):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                    self.samples.append((os.path.join(folder, fname), label))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert('RGB')
        except Exception:
            img = Image.new('RGB', (IMG_SIZE, IMG_SIZE))
        if self.transform:
            img = self.transform(img)
        return img, label

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

if os.path.exists(DATASET_DIR):
    full_dataset = TextureDataset(DATASET_DIR, transform=train_transform)
    if len(full_dataset) > 0:
        val_size   = int(0.2 * len(full_dataset))
        train_size = len(full_dataset) - val_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)
        
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        for param in model.features.parameters():
            param.requires_grad = False
            
        model.classifier[1] = nn.Linear(model.last_channel, NUM_CLASSES)
        model = model.to(DEVICE)
        
        optimizer = torch.optim.Adam(model.classifier.parameters(), lr=LR)
        criterion = nn.CrossEntropyLoss()
        
        best_val_acc = 0.0
        for epoch in range(EPOCHS):
            model.train()
            for images, labels in train_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(images)
                loss    = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
            model.eval()
            val_correct = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(DEVICE), labels.to(DEVICE)
                    outputs = model(images)
                    val_correct += (outputs.argmax(1) == labels).sum().item()
                    
            val_acc = val_correct / len(val_dataset)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), MODEL_OUTPUT)
                
        print(f"Training complete. Best val accuracy: {best_val_acc:.3f}")
else:
    print(f"Dataset directory {DATASET_DIR} not found. Please organize images into 0/, 1/, 2/ subfolders first.")
