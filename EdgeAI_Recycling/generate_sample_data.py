

import os
from PIL import Image, ImageDraw
import random

# Classes and sample count
classes = ["plastic", "paper", "metal", "glass"]
samples_per_class = 10
image_size = (96, 96)
output_dir = "data"

# Create directories
for split in ["train", "val"]:
    for cls in classes:
        dir_path = os.path.join(output_dir, split, cls)
        os.makedirs(dir_path, exist_ok=True)

# Function to generate random "recyclable" images
def generate_image(cls):
    img = Image.new("RGB", image_size, color=(255,255,255))
    draw = ImageDraw.Draw(img)
    
    # Draw random shapes/colors per class
    if cls == "plastic":
        draw.ellipse([20,20,76,76], fill=(255,0,0))  # red circle
    elif cls == "paper":
        draw.rectangle([20,20,76,76], fill=(0,255,0))  # green square
    elif cls == "metal":
        draw.polygon([(48,10),(10,86),(86,86)], fill=(192,192,192))  # gray triangle
    elif cls == "glass":
        draw.ellipse([10,10,86,86], fill=(0,0,255))  # blue circle
    return img

# Generate images
for split in ["train", "val"]:
    for cls in classes:
        for i in range(samples_per_class):
            img = generate_image(cls)
            img.save(os.path.join(output_dir, split, cls, f"{cls}_{i}.png"))

print("Sample dataset generated successfully!")
