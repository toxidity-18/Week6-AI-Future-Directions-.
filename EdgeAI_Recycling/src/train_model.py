"""
train_model.py

This script trains a lightweight image classification model on recyclable items.
The model can later be converted to TensorFlow Lite for Edge AI deployment.

Requirements:
- TensorFlow
- NumPy
- scikit-learn (for label preprocessing)
"""

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
import os

# Paths
train_dir = "data/train"
val_dir = "data/val"
model_save_path = "models/recycle_model.h5"

# Image data generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest"
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(96,96), batch_size=32, class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir, target_size=(96,96), batch_size=32, class_mode='categorical'
)

# Build model (MobileNetV2 as base)
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(96,96,3))
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(train_generator.num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=x)

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10
)

# Save trained model
os.makedirs("models", exist_ok=True)
model.save(model_save_path)
print(f"Model saved at {model_save_path}")
