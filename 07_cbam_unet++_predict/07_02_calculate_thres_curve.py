import os

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import precision_recall_curve, roc_curve, auc, average_precision_score
from tqdm import tqdm

# Device setup
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Preprocessing transformation
transform = A.Compose([
    A.Resize(512, 512),
    A.Normalize(),
    ToTensorV2(),
])

# Paths
data_dirs = {"image": "data/image", "label": "data/label"}
model_paths = [
    "models/model_1.pt",
    "models/model_2.pt",
    "models/model_3.pt"
]

# Model labels and colors for visualization
labels = ["Model 1", "Model 2", "Model 3"]
colors = ['#ff7f0e', '#1f77b4', '#2ca02c']

# Metrics storage
results = []
fig, axs = plt.subplots(2, 2, figsize=(12, 10))


def get_metrics(model_path):
    model = torch.load(model_path, map_location=device)
    model.eval()
    probabilities, ground_truths = [], []

    for image_name in tqdm(os.listdir(data_dirs["image"])):
        image_path = os.path.join(data_dirs["image"], image_name)
        label_path = os.path.join(data_dirs["label"], image_name)

        image = cv2.imread(image_path)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        if image is None or label is None:
            continue

        image = transform(image=image)['image'].unsqueeze(0).to(device).float()
        label = (cv2.resize(label, (512, 512)) > 127).astype(np.uint8)

        with torch.no_grad():
            output = model(image)
            prob_map = torch.softmax(output, dim=1).cpu().numpy()[0][1]

        probabilities.extend(prob_map.flatten())
        ground_truths.extend(label.flatten())

    return np.array(probabilities), np.array(ground_truths)


for idx, model_path in enumerate(model_paths):
    probabilities, ground_truths = get_metrics(model_path)

    thresholds = np.linspace(0, 1, 101)
    iou_scores, dice_scores = [], []

    for threshold in thresholds:
        binary_pred = (probabilities >= threshold).astype(np.uint8)
        intersection = np.logical_and(binary_pred, ground_truths).sum()
        union = np.logical_or(binary_pred, ground_truths).sum()
        iou_scores.append(intersection / union if union > 0 else 0)
        dice_scores.append(
            (2 * intersection) / (2 * intersection + binary_pred.sum() + ground_truths.sum()) if union > 0 else 0)

    best_iou, best_dice = max(iou_scores), max(dice_scores)
    ap_score = average_precision_score(ground_truths, probabilities)
    auc_score = auc(*roc_curve(ground_truths, probabilities)[:2])
    results.append(
        f"{labels[idx]} - Best IoU: {best_iou:.4f}, Best Dice: {best_dice:.4f}, AP: {ap_score:.4f}, AUC: {auc_score:.4f}\n")

    # Plot curves
    axs[0, 0].plot(thresholds, iou_scores, label=labels[idx], color=colors[idx])
    axs[0, 1].plot(thresholds, dice_scores, label=labels[idx], color=colors[idx])
    precision, recall, _ = precision_recall_curve(ground_truths, probabilities)
    fpr, tpr, _ = roc_curve(ground_truths, probabilities)
    axs[1, 0].plot(recall, precision, label=labels[idx], color=colors[idx])
    axs[1, 1].plot(fpr, tpr, label=labels[idx], color=colors[idx])

# Save results
with open("metrics_summary.txt", "w") as f:
    f.writelines(results)

# Finalize plots
for ax, title, xlabel, ylabel in zip(axs.flat,
                                     ["IoU vs Threshold", "Dice vs Threshold", "Precision-Recall Curve", "ROC Curve"],
                                     ["Threshold", "Threshold", "Recall", "False Positive Rate"],
                                     ["IoU", "Dice", "Precision", "True Positive Rate"]):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()

plt.tight_layout()
plt.savefig("evaluation_curves.png")
plt.close()
