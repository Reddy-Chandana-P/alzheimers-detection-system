# =============================================================================
# EFFICIENTNETB4 - ALZHEIMER'S DISEASE DETECTION - TRAINING SCRIPT
# =============================================================================
# Architecture : EfficientNetB4 (Transfer Learning from ImageNet)
# Dataset      : Alzheimer's MRI Dataset (4 classes)
# Input Size   : 224x224 pixels (larger than B0 for better detail)
# Classes      : Mild_Demented, Moderate_Demented, Non_Demented, Very_Mild_Demented
# Expected Acc : 90-93%
#
# Key improvements over previous models:
#   1. EfficientNetB4 vs B0:
#      - B4 has more layers, wider channels, and higher resolution scaling
#      - Better at capturing subtle brain atrophy patterns
#      - ~19M parameters vs ~5.3M in B0
#
#   2. Class Weights:
#      - Dataset is heavily imbalanced (Moderate_Demented has only ~64 samples)
#      - Class weights penalize the model more for misclassifying rare classes
#      - This significantly improves recall for minority classes
#
#   3. Larger Input (224x224 vs 128x128):
#      - More spatial detail preserved
#      - Better detection of subtle hippocampal atrophy
#      - EfficientNetB4 was designed for 380x380 — 224 is a good compromise
#
#   4. Label Smoothing:
#      - Prevents overconfident predictions
#      - Improves generalization on small datasets
# =============================================================================

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense, GlobalAveragePooling2D, Dropout,
    Input, BatchNormalization
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt

# =============================================================================
# CONFIGURATION
# =============================================================================

IMG_SIZE    = (224, 224)       # Larger input — more detail for subtle brain changes
BATCH_SIZE  = 16               # Smaller batch due to larger image size
EPOCHS      = 50               # Maximum training epochs
LR          = 0.0001           # Initial learning rate
NUM_CLASSES = 4                # Number of output classes

# Dataset paths — update these to match your local directory
TRAIN_DIR = r'C:\Users\reddy\Downloads\Alzheimer_Splitted\train'
VAL_DIR   = r'C:\Users\reddy\Downloads\Alzheimer_Splitted\val'
TEST_DIR  = r'C:\Users\reddy\Downloads\Alzheimer_Splitted\test'

# Output model file path
MODEL_SAVE_PATH = 'alzheimer_efficientnetb4_best.h5'

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
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    brightness_range=[0.8, 1.2],   # Simulate different MRI scanner intensities
    fill_mode='nearest'
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
# CLASS WEIGHTS — Fix imbalanced dataset
# =============================================================================
# The dataset has very few Moderate_Demented samples (~64 out of 6400).
# Without class weights, the model ignores this class and gets high accuracy
# by predicting the majority class — but recall for Moderate_Demented is near 0.
#
# Class weights tell the model: "A mistake on a rare class costs more."
# This forces the model to learn all classes properly.

labels = train_generator.classes  # Integer class labels for all training images

class_weights_array = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(labels),
    y=labels
)

# Convert to dictionary format required by Keras
class_weight_dict = dict(enumerate(class_weights_array))

print("\nClass weights (higher = rarer class, penalized more):")
for idx, (cls, weight) in enumerate(zip(CLASS_NAMES, class_weights_array)):
    print(f"   {cls}: {weight:.4f}")

# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================
#
#   Input (224x224x3)
#     → EfficientNetB4 Base (pretrained on ImageNet, include_top=False)
#         Uses compound scaling: depth=1.8, width=1.6, resolution=1.3
#     → GlobalAveragePooling2D
#     → BatchNormalization
#     → Dense(512, ReLU)
#     → Dropout(0.5)
#     → Dense(256, ReLU)
#     → Dropout(0.3)
#     → Dense(4, Softmax)

input_layer = Input(shape=(224, 224, 3))

# Load EfficientNetB4 base with ImageNet weights
base_model = EfficientNetB4(
    include_top=False,         # Remove ImageNet classification head
    weights='imagenet',        # Pretrained weights
    input_tensor=input_layer
)

# Freeze base model initially
base_model.trainable = False

# Custom classification head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=input_layer, outputs=output)

# Compile with label smoothing — prevents overconfident predictions
# and improves generalization on small/imbalanced datasets
model.compile(
    optimizer=Adam(learning_rate=LR),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']
)

model.summary()
print(f"\nTotal parameters   : {model.count_params():,}")

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
# PHASE 1: TRAIN CLASSIFICATION HEAD ONLY (base model frozen)
# =============================================================================

print("\n" + "="*60)
print("PHASE 1: Training classification head (EfficientNetB4 frozen)")
print("="*60)

history_phase1 = model.fit(
    train_generator,
    epochs=20,
    validation_data=val_generator,
    callbacks=callbacks,
    class_weight=class_weight_dict,   # Apply class weights here
    verbose=1
)

# =============================================================================
# PHASE 2: FINE-TUNING (unfreeze last 50 layers)
# =============================================================================

print("\n" + "="*60)
print("PHASE 2: Fine-tuning last 50 layers of EfficientNetB4")
print("="*60)

# Unfreeze last 50 layers for fine-tuning
for layer in base_model.layers[-50:]:
    if not isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = True

# Recompile with lower learning rate
model.compile(
    optimizer=Adam(learning_rate=LR / 10),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']
)

history_phase2 = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=callbacks,
    class_weight=class_weight_dict,   # Apply class weights here too
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

# Per-class metrics
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Get predictions
test_generator.reset()
y_pred_probs = model.predict(test_generator, verbose=1)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = test_generator.classes

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

# =============================================================================
# CONFUSION MATRIX
# =============================================================================

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES
)
plt.title('EfficientNetB4 - Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('efficientnetb4_confusion_matrix.png', dpi=100)
plt.show()

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
    plt.savefig('efficientnetb4_training_history.png', dpi=100)
    plt.show()

plot_history(history_phase2, 'EfficientNetB4 Fine-tuning')

print(f"\n✅ Best model saved to: {MODEL_SAVE_PATH}")
print(f"   Expected accuracy: 90-93%")
print(f"   Expected recall  : 88-92%")
