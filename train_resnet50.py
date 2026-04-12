# =============================================================================
# RESNET50 - ALZHEIMER'S DISEASE DETECTION - TRAINING SCRIPT
# =============================================================================
# Architecture : ResNet50 (Transfer Learning from ImageNet)
# Dataset      : Alzheimer's MRI Dataset (4 classes)
# Input Size   : 128x128 pixels
# Classes      : Mild_Demented, Moderate_Demented, Non_Demented, Very_Mild_Demented
# =============================================================================

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# =============================================================================
# CONFIGURATION
# =============================================================================

IMG_SIZE    = (128, 128)       # Input image dimensions
BATCH_SIZE  = 32               # Number of images per training batch
EPOCHS      = 50               # Maximum training epochs (EarlyStopping may stop earlier)
LR          = 0.0001           # Initial learning rate
NUM_CLASSES = 4                # Number of output classes

# Dataset paths — update these to match your local directory
TRAIN_DIR = r'C:\Users\reddy\Downloads\Alzheimer_Splitted\train'
VAL_DIR   = r'C:\Users\reddy\Downloads\Alzheimer_Splitted\val'
TEST_DIR  = r'C:\Users\reddy\Downloads\Alzheimer_Splitted\test'

# Output model file path
MODEL_SAVE_PATH = 'alzheimer_resnet50_best.h5'

# Class names (must match folder names in dataset)
CLASS_NAMES = ['Mild_Demented', 'Moderate_Demented', 'Non_Demented', 'Very_Mild_Demented']

# =============================================================================
# DATA GENERATORS
# =============================================================================
# ImageDataGenerator handles loading images from disk in batches.
# Training data uses augmentation to improve generalization.
# Validation and test data are only normalized (no augmentation).

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
# MODEL ARCHITECTURE
# =============================================================================
# ResNet50 uses residual (skip) connections to allow very deep networks
# to train effectively without vanishing gradients.
#
# Architecture:
#   Input (128x128x3)
#     → ResNet50 Base (pretrained on ImageNet, include_top=False)
#     → GlobalAveragePooling2D
#     → BatchNormalization
#     → Dense(512, ReLU)
#     → Dropout(0.5)
#     → Dense(256, ReLU)
#     → Dropout(0.3)
#     → Dense(4, Softmax)  ← 4 Alzheimer's classes

input_layer = Input(shape=(128, 128, 3))

# Load ResNet50 base with ImageNet weights, excluding the top classification layer
base_model = ResNet50(
    include_top=False,       # Remove the original ImageNet classification head
    weights='imagenet',      # Use pretrained ImageNet weights
    input_tensor=input_layer
)

# Freeze base model layers initially — train only the custom top layers first
# This prevents destroying the pretrained features in early training
base_model.trainable = False

# Add custom classification head
x = base_model.output
x = GlobalAveragePooling2D()(x)          # Reduce spatial dimensions to 1D
x = BatchNormalization()(x)              # Normalize activations for stable training
x = Dense(512, activation='relu')(x)    # First fully connected layer
x = Dropout(0.5)(x)                     # 50% dropout for regularization
x = Dense(256, activation='relu')(x)    # Second fully connected layer
x = Dropout(0.3)(x)                     # 30% dropout
output = Dense(NUM_CLASSES, activation='softmax')(x)  # Output: probability per class

model = Model(inputs=input_layer, outputs=output)

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=LR),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

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
# PHASE 1: TRAIN TOP LAYERS ONLY (base model frozen)
# =============================================================================

print("\n" + "="*60)
print("PHASE 1: Training top layers (base model frozen)")
print("="*60)

history_phase1 = model.fit(
    train_generator,
    epochs=20,
    validation_data=val_generator,
    callbacks=callbacks,
    verbose=1
)

# =============================================================================
# PHASE 2: FINE-TUNING (unfreeze last 50 layers of base model)
# =============================================================================
# After the top layers have learned basic features, we unfreeze the deeper
# layers of ResNet50 and fine-tune them with a very low learning rate.

print("\n" + "="*60)
print("PHASE 2: Fine-tuning (unfreezing last 50 layers)")
print("="*60)

# Unfreeze the last 50 layers of the base model
for layer in base_model.layers[-50:]:
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
# EVALUATION
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
    plt.savefig(f'resnet50_training_history.png', dpi=100)
    plt.show()

plot_history(history_phase2, 'ResNet50 Fine-tuning')

print(f"\n✅ Best model saved to: {MODEL_SAVE_PATH}")
