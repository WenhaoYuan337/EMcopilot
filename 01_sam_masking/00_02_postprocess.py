import os

import cv2
import numpy as np
from skimage.measure import label, regionprops
from tqdm import tqdm

# Define directories
mask_dir = "<MASK_DIRECTORY>"
image_dir = "<IMAGE_DIRECTORY>"
output_dir = "<OUTPUT_DIRECTORY>"
vis_dir = "<VISUALIZATION_DIRECTORY>"

# Ensure output directories exist
os.makedirs(output_dir, exist_ok=True)
os.makedirs(vis_dir, exist_ok=True)

# Intensity threshold for filtering
intensity_threshold = 60


def process_mask(mask, image):
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    masks_to_keep = np.zeros_like(mask, dtype=np.uint8)

    for contour in contours:
        single_mask = np.zeros_like(mask, dtype=np.uint8)
        cv2.drawContours(single_mask, [contour], -1, 255, thickness=cv2.FILLED)

        masked_region = cv2.bitwise_and(image, image, mask=single_mask)
        valid_pixels = masked_region[masked_region > 0]
        mean_intensity = np.mean(valid_pixels) if len(valid_pixels) > 0 else 0

        if mean_intensity > intensity_threshold:
            cv2.drawContours(masks_to_keep, [contour], -1, 255, thickness=cv2.FILLED)
    return masks_to_keep


def edge_filter(mask):
    labeled_mask = label(mask)
    filtered_mask = np.zeros_like(mask)
    boundary_margin = 5
    inner_x_min, inner_x_max = boundary_margin, 512 - boundary_margin
    inner_y_min, inner_y_max = boundary_margin, 512 - boundary_margin

    for region in regionprops(labeled_mask):
        min_row, min_col, max_row, max_col = region.bbox
        if (min_row >= inner_y_min and max_row <= inner_y_max and
                min_col >= inner_x_min and max_col <= inner_x_max):
            filtered_mask[labeled_mask == region.label] = 255
    return filtered_mask


def visualize(image, orig_mask, final_mask, vis_path):
    vis_image_orig = image.copy()
    vis_image_final = image.copy()

    mask_colored_orig = np.zeros_like(image)
    contours_orig, _ = cv2.findContours(orig_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours_orig:
        color = np.random.randint(0, 255, size=3).tolist()
        single_mask = np.zeros_like(orig_mask, dtype=np.uint8)
        cv2.drawContours(single_mask, [contour], -1, 255, thickness=cv2.FILLED)
        mask_colored_orig[single_mask > 0] = color

    overlay_orig = cv2.addWeighted(vis_image_orig, 1.0, mask_colored_orig, 0.35, 0)

    mask_colored_final = np.zeros_like(image)
    contours_final, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours_final:
        color = np.random.randint(0, 255, size=3).tolist()
        single_mask = np.zeros_like(final_mask, dtype=np.uint8)
        cv2.drawContours(single_mask, [contour], -1, 255, thickness=cv2.FILLED)
        mask_colored_final[single_mask > 0] = color

    overlay_final = cv2.addWeighted(vis_image_final, 1.0, mask_colored_final, 0.35, 0)

    vis_combined = np.hstack((overlay_orig, overlay_final))
    cv2.imwrite(vis_path, vis_combined)


for mask_filename in tqdm(os.listdir(mask_dir), desc="Processing masks"):
    if not mask_filename.endswith(('.png', '.jpg', '.jpeg')):
        continue

    mask_path = os.path.join(mask_dir, mask_filename)
    image_path = os.path.join(image_dir, mask_filename)
    if not os.path.exists(image_path):
        print(f"Warning: Missing corresponding image for {mask_filename}")
        continue

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)
    image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_LINEAR)

    processed_mask = process_mask(mask, image)
    final_mask = edge_filter(processed_mask)
    output_path = os.path.join(output_dir, mask_filename)
    cv2.imwrite(output_path, final_mask)

    vis_path = os.path.join(vis_dir, mask_filename)
    visualize(image, mask, final_mask, vis_path)


print("Processing and visualization completed.")
