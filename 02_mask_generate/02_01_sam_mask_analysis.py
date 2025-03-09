import json
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from skimage.measure import regionprops, label
from tqdm import tqdm

# Define input and output directories
input_folder = "<INPUT_DIRECTORY>"
output_folder = "<OUTPUT_DIRECTORY>"
os.makedirs(output_folder, exist_ok=True)

# Configure Matplotlib
plt.rcParams["font.family"] = "Arial"
plt.rcParams["axes.labelsize"] = 18
plt.rcParams["xtick.labelsize"] = 18
plt.rcParams["ytick.labelsize"] = 18
plt.rcParams["legend.fontsize"] = 18

# List to store all statistics
all_particles = []


def analyze_mask(mask_path, mask_name):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    labeled_mask = label(mask)
    regions = regionprops(labeled_mask)

    mask_summary = []
    mask_contours = {"particles": {}}

    for idx, region in enumerate(regions, start=1):
        contour_mask = (labeled_mask == region.label).astype(np.uint8) * 255
        contours, _ = cv2.findContours(contour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            continue

        hull = cv2.convexHull(contours[0])
        mask_contours["particles"][str(idx)] = {"contour": hull.reshape(-1, 2).tolist()}

        perimeter = cv2.arcLength(contours[0], True)
        area = region.area
        if area < 10:
            continue

        eccentricity = region.eccentricity
        aspect_ratio = region.major_axis_length / max(region.minor_axis_length, 1)
        circularity = min(4 * np.pi * area / (perimeter ** 2), 1) if perimeter > 0 else 0
        solidity = region.solidity

        mask_summary.append([idx, area, perimeter, eccentricity, aspect_ratio, circularity, solidity, mask_name])
        all_particles.append(mask_summary[-1])

    summary_df = pd.DataFrame(mask_summary, columns=["Particle_ID", "Area", "Perimeter", "Eccentricity",
                                                     "Aspect Ratio", "Circularity", "Solidity", "Mask"])
    summary_df.to_csv(os.path.join(output_folder, f"{mask_name}_mask_summary.csv"), index=False)

    with open(os.path.join(output_folder, f"{mask_name}_contours.json"), "w") as f:
        json.dump(mask_contours, f, indent=4)


mask_files = [f for f in os.listdir(input_folder) if f.endswith(".png")]
for mask_file in tqdm(mask_files, desc="Analyzing masks"):
    analyze_mask(os.path.join(input_folder, mask_file), mask_file.replace(".png", ""))

all_summary_df = pd.DataFrame(all_particles, columns=["Particle_ID", "Area", "Perimeter", "Eccentricity",
                                                      "Aspect Ratio", "Circularity", "Solidity", "Mask"])
all_summary_df.to_csv(os.path.join(output_folder, "all_mask_summary.csv"), index=False)

# Generate KDE histograms
fig, axes = plt.subplots(3, 2, figsize=(12, 14))
metrics = ["Area", "Perimeter", "Eccentricity", "Circularity", "Aspect Ratio", "Solidity"]
colors = sns.color_palette("husl", len(metrics))

for ax, metric, color in zip(axes.flatten(), metrics, colors):
    sns.histplot(all_summary_df[metric], kde=True, color=color, ax=ax, stat="density")
    ax.set_xlabel(metric, fontsize=18)
    ax.set_ylabel("Density", fontsize=18)

plt.tight_layout()
plt.savefig(os.path.join(output_folder, "overall_mask_histogram.png"))
plt.close()

print("Processing complete. Data and visualizations saved.")
