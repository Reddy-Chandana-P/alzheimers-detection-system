# =============================================================================
# EFFICIENTNETB0 + MOBILENETV2 HYBRID - ALZHEIMER'S DISEASE DETECTION
# =============================================================================
# Architecture : EfficientNetB0 + MobileNetV2 (Hybrid Ensemble)
# Dataset      : Alzheimer's MRI Dataset (4 classes)
# Input Size   : 128x128 pixels
# Classes      : Mild_Demented, Moderate_Demented, Non_Demented, Very_Mild_Demented
# Test Accuracy: 73%
#
# Why this hybrid?
#   - EfficientNetB0: Uses compound scaling (depth + width + resolution)
#     for efficient and accurate feature extraction
#   - MobileNetV2: Uses depthwise separable convolutions for lightweight,
#     fast inference — ideal for deployment
#   - Combining both gives complementary feature representations,
#     improving overall accuracy and robustness
# =============================================================================

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0, MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense, GlobalAveragePooling2D, Dropout, Input,
    BatchNormalization, Concatenate
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# =============================================================================
# CONFIGURATION
# =============================================================================

IMG_SIZE    = (128, 128)       # Input image dimensions
BATCH_SIZE  = 32               # Number of images per training batch
EPOCHS      = 50               # Maximum training epochs
LR          = 0.0001           # Initial learning rate
NUM_CLASSES = 4                # Number of output classes

# Dataset paths — update these to match your local directory
TRAIN_DIR = r'C:\Users\reddy\Downloads\Alzheimer_Splitted\train'
VAL_DIR   = r'C:\Users\reddy\Downloads\Alzheimer_Splitted\val'
TEST_DIR  = r'C:\Users\reddy\Downloads\Alzheimer_Splitted\test'

# Output model file path
MODEL_SAVE_PATH = 'alzheimer_128_best.h5'

# Class names (must match folder names in dataset)
CLASS_NAMES = ['Mild_Demented', 'Moderate_Demented', 'Non_Demented', 'Very_Mild_Demented']

# =============================================================================
# DATA GENERATORS
# =============================================================================
# Training data uses augmentation to improve generalization.
# Validation and test data are only normalized — no augmentation.

train_datagen = ImageDataGenerator(
    rescale=1./255,          # Normalize pixel values to [0, 1]
    shear_range=0.2,         # Random shear transformation
    zoom_range=0.2,          # Random zoom
    horizontal_flip=True,    # Random horizontal flip
    rotation_range=15,       # Random rotation up to 15 degrees
    width_shift_range=0.1,   # Random horizontal shift
    height_shift_range=0.1   # Random vertical shift
)

val_test_datagen = ImageDataGenerator(
    rescale=1./255           # Only normalize — no augmentation for val/test
)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True,
    seed=42
)

val_generator = val_test_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

test_generator = val_test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

print(f"\nTraining samples   : {train_generator.samples}")
print(f"Validation samples : {val_generator.samples}")
print(f"Test samples       : {test_generator.samples}")
print(f"Classes            : {list(train_generator.class_indices.keys())}")

# =============================================================================
# MODEL ARCHITECTURE — HYBRID EfficientNetB0 + MobileNetV2
# =============================================================================
# Both models share the same input image.
# Their feature outputs are concatenated before the classification head.
#
#   Input (128x128x3)
#       ├── EfficientNetB0 Base → GlobalAveragePooling2D → features_1
#       └── MobileNetV2 Base   → GlobalAveragePooling2D → features_2
#                                         ↓
#                               Concatenate([features_1, features_2])
#                                         ↓
#                               BatchNormalization
#                                         ↓
#                               Dense(512, ReLU)
#                                         ↓
#                               Dropout(0.5)
#                                         ↓
#                               Dense(256, ReLU)
#                                         ↓
#                               Dropout(0.3)
#                                         ↓
#                               Dense(4, Softmax)  ← 4 Alzheimer's classes

# Shared input layer — both models receive the same image
input_layer = Input(shape=(128, 128, 3))

# --- EfficientNetB0 Branch ---
# EfficientNetB0 uses compound scaling to balance network depth, width,
# and resolution for efficient and accurate feature extraction
efficientnet_base = EfficientNetB0(
    include_top=False,       # Remove ImageNet classification head
    weights='imagenet',      # Use pretrained ImageNet weights
    input_tensor=input_layer
)
efficientnet_base.trainable = False  # Freeze during initial training
eff_features = GlobalAveragePooling2D()(efficientnet_base.output)

# --- MobileNetV2 Branch ---
# MobileNetV2 uses inverted residuals and linear bottlenecks for
# lightweight, fast inference — ideal for deployment
mobilenet_base = MobileNetV2(
    include_top=False,
    weights='imagenet',
    input_tensor=input_layer,
    alpha=1.0                # Width multiplier — 1.0 = full size
)
mobilenet_base.trainable = False  # Freeze during initial training
mob_features = GlobalAveragePooling2D()(mobilenet_base.output)

# --- Merge both feature vectors ---
# Concatenation combines the complementary features from both architectures
merged = Concatenate()([eff_features, mob_features])

# --- Classification Head ---
x = BatchNormalization()(merged)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)                     # 50% dropout for regularization
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=input_layer, outputs=output)

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=LR),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()
print(f"\nTotal parameters   : {model.count_params():,}")

# =============================================================================
# CALLBACKS
# =============================================================================

callbacks = [
    # Save the best model based on validation accuracy
    ModelCheckpoint(
        MODEL_SAVE_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    # Stop training if validation loss doesn't improve for 10 epochs
    EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    # Reduce learning rate when validation loss plateaus
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
]

# =============================================================================
# PHASE 1: TRAIN CLASSIFICATION HEAD ONLY (both base models frozen)
# =============================================================================

print("\n" + "="*60)
print("PHASE 1: Training classification head (base models frozen)")
print("="*60)

history_phase1 = model.fit(
    train_generator,
    epochs=20,
    validation_data=val_generator,
    callbacks=callbacks,
    verbose=1
)

# =============================================================================
# PHASE 2: FINE-TUNING (unfreeze top layers of both base models)
# =============================================================================
# After the classification head has learned, we unfreeze the top layers
# of both base models and fine-tune them together with a lower learning rate.

print("\n" + "="*60)
print("PHASE 2: Fine-tuning top layers of both base models")
print("="*60)

# Unfreeze last 30 layers of EfficientNetB0
for layer in efficientnet_base.layers[-30:]:
    layer.trainable = True

# Unfreeze last 30 layers of MobileNetV2
for layer in mobilenet_base.layers[-30:]:
    layer.trainable = True

# Recompile with a much lower learning rate for fine-tuning
model.compile(
    optimizer=Adam(learning_rate=LR / 10),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history_phase2 = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=callbacks,
    verbose=1
)

# =============================================================================
# EVALUATION ON TEST SET
# =============================================================================

print("\n" + "="*60)
print("EVALUATING ON TEST SET")
print("="*60)

test_loss, test_accuracy = model.evaluate(test_generator, verbose=1)
print(f"\nTest Loss     : {test_loss:.4f}")
print(f"Test Accuracy : {test_accuracy * 100:.2f}%")

# =============================================================================
# PLOT TRAINING HISTORY
# =============================================================================

def plot_history(history, title):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history.history['accuracy'], label='Train Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Val Accuracy')
    ax1.set_title(f'{title} - Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(history.history['loss'], label='Train Loss')
    ax2.plot(history.history['val_loss'], label='Val Loss')
    ax2.set_title(f'{title} - Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('efficientnet_mobilenet_training_history.png', dpi=100)
    plt.show()

plot_history(history_phase2, 'EfficientNetB0 + MobileNetV2 Fine-tuning')

print(f"\n✅ Best model saved to: {MODEL_SAVE_PATH}")
