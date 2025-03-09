import os

import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# Directories (replace with actual paths)
image_dir = "<IMAGE_DIRECTORY>"
output_dir = "<OUTPUT_DIRECTORY>"
os.makedirs(output_dir, exist_ok=True)

# Model configuration
sam_checkpoint = "<SAM_MODEL_CHECKPOINT>"
model_type = "vit_h"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the SAM model
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device)
sam.to(device=device)

# Initialize mask generator
mask_generator_ = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=32,
    points_per_batch=32,
    pred_iou_thresh=0.01,
    stability_score_thresh=0.9,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=0
)

# Define mask size constraints
min_mask_region_area = 0
max_mask_region_area = 1000

# Process all images
image_files = [f for f in os.listdir(image_dir) if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))]

for image_file in image_files:
    image_path = os.path.join(image_dir, image_file)
    image = cv2.imread(image_path)

    if image is None:
        print(f"Warning: Unable to read {image_path}")
        continue

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Generate masks
    masks = mask_generator_.generate(image)
    masks = [mask for mask in masks if min_mask_region_area <= mask["area"] <= max_mask_region_area]

    if len(masks) == 0:
        print(f"Warning: {image_file} - No valid masks found, skipping.")
        continue

    # Create binary mask
    mask_shape = masks[0]['segmentation'].shape
    binary_mask = np.zeros(mask_shape, dtype=np.uint8)

    for ann in masks:
        binary_mask[ann['segmentation']] = 255

    # Save binary mask
    output_path = os.path.join(output_dir, image_file)
    cv2.imwrite(output_path, binary_mask)
    print(f"Processed: {image_file} -> {output_path}")

print("Processing complete.")
