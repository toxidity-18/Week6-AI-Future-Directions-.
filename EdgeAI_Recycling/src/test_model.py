

import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

tflite_model_path = "tflite_models/recycle_model.tflite"
sample_image_path = "data/val/plastic/plastic_0.png"

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load and preprocess image
img = image.load_img(sample_image_path, target_size=(96,96))
img_array = image.img_to_array(img)/255.0
img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

# Run inference
interpreter.set_tensor(input_details[0]['index'], img_array)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])

pred_class = np.argmax(output_data)
print(f"Predicted class index: {pred_class}")
