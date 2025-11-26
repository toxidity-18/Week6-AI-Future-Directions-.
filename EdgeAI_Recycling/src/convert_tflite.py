

import tensorflow as tf
import os

model_path = "models/recycle_model.h5"
tflite_model_path = "tflite_models/recycle_model.tflite"

# Load Keras model
model = tf.keras.models.load_model(model_path)

# Convert to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save TFLite model
os.makedirs("tflite_models", exist_ok=True)
with open(tflite_model_path, "wb") as f:
    f.write(tflite_model)

print(f"TFLite model saved at {tflite_model_path}")
