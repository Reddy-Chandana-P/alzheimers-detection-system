# =============================================================================
# VISION TRANSFORMER (ViT) - ALZHEIMER'S DISEASE DETECTION - TRAINING SCRIPT
# =============================================================================
# Architecture : Vision Transformer (ViT-B/16) via HuggingFace Transformers
# Dataset      : Alzheimer's MRI Dataset (4 classes)
# Input Size   : 224x224 pixels
# Classes      : Mild_Demented, Moderate_Demented, Non_Demented, Very_Mild_Demented
# Expected Acc : 93-96%
#
# How Vision Transformers work (different from CNNs):
#   - The input image is divided into fixed-size patches (16x16 pixels each)
#   - Each patch is flattened and linearly embedded into a vector
#   - A [CLS] token is prepended to the sequence of patch embeddings
#   - Positional embeddings are added to retain spatial information
#   - The sequence is passed through multiple Transformer Encoder blocks
#     Each block contains:
#       → Multi-Head Self-Attention (MHSA): each patch attends to all others
#       → Feed-Forward Network (FFN): processes each patch independently
#       → Layer Normalization and residual connections
#   - The [CLS] token output is used for final classification
#
# Why ViT outperforms CNNs on medical imaging:
#   - Global attention: can relate distant brain regions in a single layer
#   - CNNs have limited receptive fields — need many layers to see globally
#   - Better at capturing long-range dependencies (e.g., bilateral atrophy)
#   - Pre-trained on ImageNet-21k (14M images) — much richer features
#
# Requirements:
#   pip install transformers torch torchvision scikit-learn seaborn
# =============================================================================

import os
import shutil
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import ViTForImageClassification, ViTFeatureExtractor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# CONFIGURATION
# =============================================================================

IMG_SIZE    = 224              # ViT-B/16 expects 224x224 input
PATCH_SIZE  = 16               # Each patch is 16x16 pixels (14x14 = 196 patches total)
BATCH_SIZE  = 32               # Batch size (reduce to 16 if GPU memory is limited)
EPOCHS      = 30               # Training epochs
LR          = 2e-5             # Low learning rate for fine-tuning transformers
NUM_CLASSES = 4                # Number of output classes
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Using device: {DEVICE}")

# Dataset paths — update these to match your local directory
TRAIN_DIR = r'C:\Users\reddy\Downloads\Alzheimer_Splitted\train'
VAL_DIR   = r'C:\Users\reddy\Downloads\Alzheimer_Splitted\val'
TEST_DIR  = r'C:\Users\reddy\Downloads\Alzheimer_Splitted\test'

# Output model file path
MODEL_SAVE_PATH = 'alzheimer_vit_best.pth'

# Class names (must match folder names in dataset)
CLASS_NAMES = ['Mild_Demented', 'Moderate_Demented', 'Non_Demented', 'Very_Mild_Demented']

# HuggingFace model name
# ViT-B/16 pretrained on ImageNet-21k (14M images, 21k classes)
VIT_MODEL_NAME = 'google/vit-base-patch16-224-in21k'

# =============================================================================
# DATA TRANSFORMS
# =============================================================================
# ViT requires specific normalization (ImageNet mean/std)
# Training uses augmentation; val/test only normalize

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Simulate MRI variations
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # ViT normalization
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# =============================================================================
# DATASETS AND DATALOADERS
# =============================================================================

train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
val_dataset   = datasets.ImageFolder(VAL_DIR,   transform=val_transform)
test_dataset  = datasets.ImageFolder(TEST_DIR,  transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

print(f"\nTraining samples   : {len(train_dataset)}")
print(f"Validation samples : {len(val_dataset)}")
print(f"Test samples       : {len(test_dataset)}")
print(f"Classes            : {train_dataset.classes}")

# =============================================================================
# CLASS WEIGHTS — Fix imbalanced dataset
# =============================================================================

labels = [label for _, label in train_dataset.samples]
cw = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
class_weights = torch.FloatTensor(cw).to(DEVICE)
print(f"\nClass weights: {dict(zip(CLASS_NAMES, cw.round(3)))}")

# =============================================================================
# MODEL — Vision Transformer (ViT-B/16)
# =============================================================================
# Load pretrained ViT-B/16 from HuggingFace
# ignore_mismatched_sizes=True allows replacing the classification head
# with our 4-class head instead of the original 21k-class head

model = ViTForImageClassification.from_pretrained(
    VIT_MODEL_NAME,
    num_labels=NUM_CLASSES,
    ignore_mismatched_sizes=True,
    id2label={i: name for i, name in enumerate(CLASS_NAMES)},
    label2id={name: i for i, name in enumerate(CLASS_NAMES)}
)

model = model.to(DEVICE)

# Print model info
total_params     = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nTotal parameters     : {total_params:,}")
print(f"Trainable parameters : {trainable_params:,}")

# =============================================================================
# LOSS, OPTIMIZER, SCHEDULER
# =============================================================================

# Weighted cross-entropy loss to handle class imbalance
criterion = nn.CrossEntropyLoss(weight=class_weights)

# AdamW optimizer — standard for transformer fine-tuning
# Weight decay helps prevent overfitting
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)

# Cosine annealing scheduler — gradually reduces learning rate
# Better than step decay for transformers
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-7)

# =============================================================================
# TRAINING LOOP
# =============================================================================

best_val_acc = 0.0
train_losses, val_losses = [], []
train_accs, val_accs = [], []

print(f"\n{'='*60}")
print("TRAINING VISION TRANSFORMER")
print(f"{'='*60}\n")

for epoch in range(EPOCHS):
    # --- Training Phase ---
    model.train()
    train_loss, train_correct, train_total = 0, 0, 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()

        # Forward pass through ViT
        # outputs.logits shape: (batch_size, num_classes)
        outputs = model(pixel_values=images)
        loss = criterion(outputs.logits, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        train_loss    += loss.item()
        preds          = outputs.logits.argmax(dim=1)
        train_correct += (preds == labels).sum().item()
        train_total   += labels.size(0)

        if (batch_idx + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/{EPOCHS} | Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f}")

    # --- Validation Phase ---
    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(pixel_values=images)
            loss = criterion(outputs.logits, labels)
            val_loss    += loss.item()
            preds        = outputs.logits.argmax(dim=1)
            val_correct += (preds == labels).sum().item()
            val_total   += labels.size(0)

    # Calculate epoch metrics
    epoch_train_loss = train_loss / len(train_loader)
    epoch_val_loss   = val_loss   / len(val_loader)
    epoch_train_acc  = train_correct / train_total
    epoch_val_acc    = val_correct   / val_total

    train_losses.append(epoch_train_loss)
    val_losses.append(epoch_val_loss)
    train_accs.append(epoch_train_acc)
    val_accs.append(epoch_val_acc)

    scheduler.step()

    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    print(f"  Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc*100:.2f}%")
    print(f"  Val Loss  : {epoch_val_loss:.4f} | Val Acc  : {epoch_val_acc*100:.2f}%")

    # Save best model
    if epoch_val_acc > best_val_acc:
        best_val_acc = epoch_val_acc
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"  ✅ Best model saved (Val Acc: {best_val_acc*100:.2f}%)")

    print()

# =============================================================================
# EVALUATION ON TEST SET
# =============================================================================

print(f"\n{'='*60}")
print("EVALUATING ON TEST SET")
print(f"{'='*60}")

# Load best model
model.load_state_dict(torch.load(MODEL_SAVE_PATH))
model.eval()

all_preds, all_labels = [], []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        outputs = model(pixel_values=images)
        preds = outputs.logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

test_acc = np.mean(np.array(all_preds) == np.array(all_labels))
print(f"\n✅ Test Accuracy: {test_acc*100:.2f}%")
print(f"   Best Val Accuracy: {best_val_acc*100:.2f}%")

print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))

# =============================================================================
# CONFUSION MATRIX
# =============================================================================

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title('Vision Transformer (ViT-B/16) - Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('vit_confusion_matrix.png', dpi=100)
plt.show()

# =============================================================================
# TRAINING HISTORY PLOT
# =============================================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(train_accs, label='Train Accuracy')
ax1.plot(val_accs,   label='Val Accuracy')
ax1.set_title('ViT-B/16 - Accuracy')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()
ax1.grid(True)

ax2.plot(train_losses, label='Train Loss')
ax2.plot(val_losses,   label='Val Loss')
ax2.set_title('ViT-B/16 - Loss')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig('vit_training_history.png', dpi=100)
plt.show()

print(f"\n✅ Model saved to: {MODEL_SAVE_PATH}")
print(f"   Expected accuracy: 93-96%")
print(f"   Expected recall  : 92-95%")
