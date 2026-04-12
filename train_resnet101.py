# =============================================================================
# RESNET101 - ALZHEIMER'S DISEASE DETECTION - TRAINING SCRIPT
# =============================================================================
# Architecture : ResNet101 (Transfer Learning from ImageNet)
# Dataset      : Alzheimer's MRI Dataset (4 classes)
# Input Size   : 128x128 pixels
# Classes      : Mild_Demented, Moderate_Demented, Non_Demented, Very_Mild_Demented
#
# ResNet101 vs ResNet50:
#   - ResNet101 has 101 layers vs ResNet50's 50 layers
#   - More parameters (~44M vs ~25M) — higher capacity
#   - Better at capturing complex patterns in medical images
#   - Slower to train but potentially higher accuracy
# =============================================================================

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet101
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
BATCH_SIZE  = 16               # Smaller batch size for larger model (memory)
EPOCHS      = 50               # Maximum training epochs
LR          = 0.0001           # Initial learning rate
NUM_CLASSES = 4                # Number of output classes

# Dataset paths — update these to match your local directory
TRAIN_DIR = r'C:\Users\reddy\Downloads\Alzheimer_Splitted\train'
VAL_DIR   = r'C:\Users\reddy\Downloads\Alzheimer_Splitted\val'
TEST_DIR  = r'C:\Users\reddy\Downloads\Alzheimer_Splitted\test'

# Output model file path
MODEL_SAVE_PATH = 'alzheimer_resnet101_best.h5'

# Class names (must match folder names in dataset)
CLASS_NAMES = ['Mild_Demented', 'Moderate_Demented', 'Non_Demented', 'Very_Mild_Demented']

# =============================================================================
# DATA GENERATORS
# =============================================================================

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=[0.8, 1.2]   # Extra augmentation for larger model
)

val_test_datagen = ImageDataGenerator(
    rescale=1./255
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
# ResNet101 uses the same residual block concept as ResNet50 but with
# more layers, allowing it to learn more complex and abstract features.
#
# Architecture:
#   Input (128x128x3)
#     → ResNet101 Base (pretrained on ImageNet, include_top=False)
#     → GlobalAveragePooling2D
#     → BatchNormalization
#     → Dense(512, ReLU)
#     → Dropout(0.5)
#     → Dense(256, ReLU)
#     → Dropout(0.3)
#     → Dense(4, Softmax)  ← 4 Alzheimer's classes

input_layer = Input(shape=(128, 128, 3))

# Load ResNet101 base with ImageNet weights
base_model = ResNet101(
    include_top=False,       # Remove the original ImageNet classification head
    weights='imagenet',      # Use pretrained ImageNet weights
    input_tensor=input_layer
)

# Freeze all base model layers initially
base_model.trainable = False

# Add custom classification head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=input_layer, outputs=output)

# Compile
model.compile(
    optimizer=Adam(learning_rate=LR),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()
print(f"\nTotal parameters   : {model.count_params():,}")
print(f"Trainable params   : {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")

# =============================================================================
# CALLBACKS
# =============================================================================

callbacks = [
    ModelCheckpoint(
        MODEL_SAVE_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
]

# =============================================================================
# PHASE 1: TRAIN TOP LAYERS ONLY
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
# PHASE 2: FINE-TUNING (unfreeze last 80 layers)
# =============================================================================
# ResNet101 has more layers so we unfreeze more (80 vs 50 for ResNet50)

print("\n" + "="*60)
print("PHASE 2: Fine-tuning (unfreezing last 80 layers)")
print("="*60)

for layer in base_model.layers[-80:]:
    layer.trainable = True

# Recompile with lower learning rate
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
    plt.savefig('resnet101_training_history.png', dpi=100)
    plt.show()

plot_history(history_phase2, 'ResNet101 Fine-tuning')

print(f"\n✅ Best model saved to: {MODEL_SAVE_PATH}")
