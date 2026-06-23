"""
Acne Severity Classifier — Transfer Learning
Trains MobileNetV2 on labeled acne images.
Run this ONCE to produce acne_model.pth.
This file is separate from the live app — you only run it during training.
"""

import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image
import numpy as np


# ── Configuration ────────────────────────────────────────────────────────────
DATASET_DIR  = 'dataset'        # folder containing your downloaded images
MODEL_OUTPUT = 'app/processing/acne_model.pth'  # where to save the trained model
NUM_CLASSES  = 4                # 0=clear, 1=mild, 2=moderate, 3=severe
EPOCHS       = 15
BATCH_SIZE   = 16
LR           = 0.001
IMG_SIZE     = 224
DEVICE       = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Training on: {DEVICE}")


# ── Dataset class ─────────────────────────────────────────────────────────────
class AcneDataset(Dataset):
    """
    Loads images from folder structure:
      dataset/train/0/*.jpg  (clear)
      dataset/train/1/*.jpg  (mild)
      etc.
    """
    def __init__(self, root_dir, transform=None):
        self.samples   = []
        self.transform = transform

        for label in range(NUM_CLASSES):
            folder = os.path.join(root_dir, str(label))
            if not os.path.exists(folder):
                print(f"Warning: folder {folder} not found, skipping")
                continue
            for fname in os.listdir(folder):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                    self.samples.append((os.path.join(folder, fname), label))

        print(f"Found {len(self.samples)} images in {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert('RGB')
        except Exception:
            # Return a blank image if file is corrupt
            img = Image.new('RGB', (IMG_SIZE, IMG_SIZE))
        if self.transform:
            img = self.transform(img)
        return img, label


# ── Transforms ────────────────────────────────────────────────────────────────
# Training: augment to simulate real-world variation
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],   # ImageNet mean
                         [0.229, 0.224, 0.225])    # ImageNet std
])

# Validation: no augmentation, just resize and normalise
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


# ── Load data ─────────────────────────────────────────────────────────────────
train_dir = os.path.join(DATASET_DIR, 'train')
val_dir   = os.path.join(DATASET_DIR, 'val')

# If no val/ folder, split train 80/20 automatically
if os.path.exists(val_dir):
    train_dataset = AcneDataset(train_dir, transform=train_transform)
    val_dataset   = AcneDataset(val_dir,   transform=val_transform)
else:
    print("No val/ folder found — splitting train 80/20 automatically")
    full_dataset  = AcneDataset(train_dir, transform=train_transform)
    val_size      = int(0.2 * len(full_dataset))
    train_size    = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


# ── Build model ───────────────────────────────────────────────────────────────
# MobileNetV2 pretrained on ImageNet — already knows edges, shapes, textures
model = models.mobilenet_v2(pretrained=True)

# Freeze all feature layers — only train the final classifier
for param in model.features.parameters():
    param.requires_grad = False

# Replace the final layer for our 4-class problem
model.classifier[1] = nn.Linear(model.last_channel, NUM_CLASSES)
model = model.to(DEVICE)

# Only optimise the new classifier layer (much faster)
optimizer = torch.optim.Adam(model.classifier.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)


# ── Training loop ─────────────────────────────────────────────────────────────
best_val_acc = 0.0

for epoch in range(EPOCHS):
    # --- Training phase ---
    model.train()
    train_loss = 0.0
    train_correct = 0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss    += loss.item() * images.size(0)
        train_correct += (outputs.argmax(1) == labels).sum().item()

    # --- Validation phase ---
    model.eval()
    val_loss = 0.0
    val_correct = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs     = model(images)
            loss        = criterion(outputs, labels)
            val_loss    += loss.item() * images.size(0)
            val_correct += (outputs.argmax(1) == labels).sum().item()

    train_acc = train_correct / len(train_dataset)
    val_acc   = val_correct   / len(val_dataset)
    scheduler.step()

    print(f"Epoch {epoch+1:02d}/{EPOCHS} | "
          f"Train loss: {train_loss/len(train_dataset):.4f} acc: {train_acc:.3f} | "
          f"Val loss: {val_loss/len(val_dataset):.4f} acc: {val_acc:.3f}")

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), MODEL_OUTPUT)
        print(f"  → Saved best model (val acc: {val_acc:.3f})")

print(f"\nTraining complete. Best val accuracy: {best_val_acc:.3f}")
print(f"Model saved to: {MODEL_OUTPUT}")
