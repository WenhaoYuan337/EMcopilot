import json
import os
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from skimage.measure import regionprops, label
from tqdm import tqdm

# Define input and output directories
json_folder = "<JSON_DIRECTORY>"
output_folder = "<OUTPUT_DIRECTORY>"
summary_folder = os.path.join(output_folder, "summary")
masks_folder = os.path.join(output_folder, "masks")
os.makedirs(summary_folder, exist_ok=True)
os.makedirs(masks_folder, exist_ok=True)

# Configuration parameters
NUM_NEW_MASKS_PER_OLD = 100
IMAGE_SIZE = 512
SHIFT_RANGE = 50
MIN_AREA = 20
MIN_PERIMETER = 15
BOUNDARY_MARGIN = 5

# Configure Matplotlib
plt.rcParams["font.family"] = "Arial"
plt.rcParams["axes.labelsize"] = 18
plt.rcParams["xtick.labelsize"] = 18
plt.rcParams["ytick.labelsize"] = 18
plt.rcParams["legend.fontsize"] = 18

# Read all JSON files
json_files = [f for f in os.listdir(json_folder) if f.endswith("_contours.json")]

# Data storage
all_particle_data = []

# Process each mask
for json_file in tqdm(json_files, desc="Generating new masks"):
    json_path = os.path.join(json_folder, json_file)
    mask_name = json_file.replace("_contours.json", "")

    with open(json_path, "r") as f:
        data = json.load(f)

    original_contours = [np.array(p["contour"], dtype=np.int32) for p in data["particles"].values()]
    num_original_particles = len(original_contours)
    new_particle_data = []

    for new_mask_idx in range(NUM_NEW_MASKS_PER_OLD):
        new_mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
        placed_contours = []

        for i in range(int(1.25 * num_original_particles)):
            if i < num_original_particles:
                contour = original_contours[i]
                center_offset = np.random.randint(-SHIFT_RANGE, SHIFT_RANGE + 1, size=(1, 2))
            else:
                contour = random.choice(original_contours)
                center_offset = np.random.randint(0, IMAGE_SIZE, size=(1, 2))

            angle = random.uniform(0, 360)
            rotation_matrix = cv2.getRotationMatrix2D(tuple(np.mean(contour, axis=0)), angle, 1.0)
            rotated_contour = cv2.transform(contour.reshape(-1, 1, 2), rotation_matrix).reshape(-1, 2)
            new_contour = rotated_contour + center_offset
            new_contour = np.clip(new_contour, 0, IMAGE_SIZE - 1).astype(np.int32)

            if any(cv2.pointPolygonTest(existing.reshape(-1, 1, 2), tuple(map(int, new_contour[0])), measureDist=False) >= 0
                   for existing in placed_contours):
                continue

            placed_contours.append(new_contour)
            cv2.drawContours(new_mask, [new_contour], -1, (255), thickness=-1)

        # Remove edge particles
        labeled_mask = label(new_mask)
        filtered_mask = np.zeros_like(new_mask)

        for region in regionprops(labeled_mask):
            min_row, min_col, max_row, max_col = region.bbox
            if (min_row >= BOUNDARY_MARGIN and max_row <= IMAGE_SIZE - BOUNDARY_MARGIN and
                    min_col >= BOUNDARY_MARGIN and max_col <= IMAGE_SIZE - BOUNDARY_MARGIN):
                filtered_mask[labeled_mask == region.label] = 255

        labeled_mask = label(filtered_mask)
        regions = regionprops(labeled_mask)

        for idx, region in enumerate(regions, start=1):
            area = region.area
            perimeter = region.perimeter
            if area < MIN_AREA or perimeter < MIN_PERIMETER:
                continue

            eccentricity = region.eccentricity
            aspect_ratio = region.major_axis_length / max(region.minor_axis_length, 1)
            circularity = min(4 * np.pi * area / (perimeter ** 2), 1) if perimeter > 0 else 0
            solidity = region.solidity

            new_particle_data.append(
                [idx, mask_name, new_mask_idx, area, perimeter, eccentricity, aspect_ratio, circularity, solidity])
            all_particle_data.append(new_particle_data[-1])

        new_mask_path = os.path.join(masks_folder, f"{mask_name}_{new_mask_idx:03d}.png")
        cv2.imwrite(new_mask_path, filtered_mask)

    df = pd.DataFrame(new_particle_data, columns=["Particle_ID", "Original_Mask", "New_Mask_Index", "Area", "Perimeter",
                                                  "Eccentricity", "Aspect Ratio", "Circularity", "Solidity"])
    df.to_csv(os.path.join(summary_folder, f"{mask_name}_gen_summary.csv"), index=False)

all_df = pd.DataFrame(all_particle_data, columns=["Particle_ID", "Original_Mask", "New_Mask_Index", "Area", "Perimeter",
                                                  "Eccentricity", "Aspect Ratio", "Circularity", "Solidity"])
all_df.to_csv(os.path.join(summary_folder, "overall_gen_summary.csv"), index=False)

# Generate KDE histograms
fig, axes = plt.subplots(3, 2, figsize=(12, 14))
metrics = ["Area", "Perimeter", "Eccentricity", "Circularity", "Aspect Ratio", "Solidity"]
colors = sns.color_palette("husl", len(metrics))

for ax, metric, color in zip(axes.flatten(), metrics, colors):
    sns.histplot(all_df[metric], kde=True, color=color, ax=ax, stat="density")
    ax.set_xlabel(metric, fontsize=18)
    ax.set_ylabel("Density", fontsize=18)

plt.tight_layout()
plt.savefig(os.path.join(summary_folder, "overall_gen_summary.png"))
plt.close()

print("Processing complete. All new masks generated and analyzed.")
