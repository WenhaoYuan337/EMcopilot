import os

import cv2
import numpy as np
import pandas as pd

# Input directories
ml_mask_dir = "segmentation_results/ml/mask"
em_mask_dir = "segmentation_results/em/mask"
gt_mask_dir = "ground_truth/mask"

# Output files
output_em_gt_csv = "results/em_gt.csv"
output_ml_gt_csv = "results/ml_gt.csv"
output_counts_csv = "results/counts.csv"

os.makedirs("results", exist_ok=True)

tolerance = 10  # Allowed centroid deviation


def get_centroid(contour):
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return None
    return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))


def resize_image(img, size=(512, 512)):
    return cv2.resize(img, size, interpolation=cv2.INTER_NEAREST)


def find_contours(mask_path, resize=False):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return []
    if resize:
        mask = resize_image(mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [{"contour": c, "area": cv2.contourArea(c), "centroid": get_centroid(c)} for c in contours if
            get_centroid(c)]


def analyze_masks(gt_dir, pred_dir, output_csv):
    results = []
    for file_name in os.listdir(gt_dir):
        gt_path = os.path.join(gt_dir, file_name)
        pred_path = os.path.join(pred_dir, file_name)
        if not os.path.exists(gt_path):
            continue

        gt_objects = find_contours(gt_path, resize=True)
        pred_objects = find_contours(pred_path) if os.path.exists(pred_path) else []
        matched_pred, matched_gt = set(), set()

        for i, pred_obj in enumerate(pred_objects):
            pred_centroid = np.array(pred_obj["centroid"])
            best_match, min_dist = None, float("inf")

            for j, gt_obj in enumerate(gt_objects):
                gt_centroid = np.array(gt_obj["centroid"])
                dist = np.linalg.norm(pred_centroid - gt_centroid)
                if dist < tolerance and dist < min_dist:
                    min_dist, best_match = dist, j

            if best_match is not None:
                results.append([file_name, pred_obj["area"], gt_objects[best_match]["area"], "Matched"])
                matched_pred.add(i)
                matched_gt.add(best_match)

        results.extend(
            [[file_name, obj["area"], "", "Pred-only"] for i, obj in enumerate(pred_objects) if i not in matched_pred])
        results.extend(
            [[file_name, "", obj["area"], "GT-only"] for j, obj in enumerate(gt_objects) if j not in matched_gt])

    pd.DataFrame(results, columns=["Image", "Pred_Area", "GT_Area", "Category"]).to_csv(output_csv, index=False)


analyze_masks(gt_mask_dir, em_mask_dir, output_em_gt_csv)
analyze_masks(gt_mask_dir, ml_mask_dir, output_ml_gt_csv)

print("Processing complete. Results saved.")