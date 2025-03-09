import os

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Preprocessing transformation
transform = A.Compose([
    A.Resize(512, 512),
    A.Normalize(),
    ToTensorV2(),
])

# Define directories
image_dir = "<INPUT_DIRECTORY>"
output_dir = "<OUTPUT_DIRECTORY>"
model_path = "<MODEL_DIRECTORY>"

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Load model
model = torch.load(model_path, map_location=device)
model.eval()

# Process images for prediction
for image_name in os.listdir(image_dir):
    image_path = os.path.join(image_dir, image_name)
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_name}")
        continue

    # Preprocess image
    image_resized = cv2.resize(image, (512, 512))
    input_tensor = transform(image=image_resized)['image'].unsqueeze(0).to(device).float()

    # Predict mask
    with torch.no_grad():
        output = model(input_tensor).cpu().numpy()[0].argmax(0)

    # Generate binary mask
    binary_mask = (output > 0).astype(np.uint8) * 255

    # Save output
    output_path = os.path.join(output_dir, image_name)
    cv2.imwrite(output_path, binary_mask)
    print(f"Saved binary mask: {output_path}")

print("Prediction complete.")
