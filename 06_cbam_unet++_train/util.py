import os
import time

import numpy as np
import pandas as pd
import torch
import tqdm

os.environ['MPLBACKEND'] = 'Agg'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from segmentation_models_pytorch import losses
import random


def train_model(model, model_name, train_loader, test_loader, device, epochs, results_dir):
    start_time = time.time()

    if len(test_loader) == 0:
        raise ValueError("Test loader is empty. Please check the test dataset.")

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=3e-4, weight_decay=5e-4)
    loss_fl = losses.FocalLoss(mode='multiclass', alpha=0.25)
    loss_jd = losses.JaccardLoss(mode='multiclass')

    os.makedirs(results_dir, exist_ok=True)
    log_path = os.path.join(results_dir, "log.csv")
    best_model_path = os.path.join(results_dir, "best_model.pt")
    plot_path = os.path.join(results_dir, "metrics.png")


    with open(log_path, "w") as f:
        f.write("epoch,phase,loss,mPA,mDice,mIoU,recall,precision\n")

    metrics = {
        "train_loss": [], "test_loss": [],
        "train_mPA": [], "test_mPA": [],
        "train_mDice": [], "test_mDice": [],
        "train_mIoU": [], "test_mIoU": [],
        "train_recall": [], "test_recall": [],
        "train_precision": [], "test_precision": []
    }

    best_mIoU = 0
    best_mDice = 0
    best_mPA = 0
    best_recall = 0
    best_precision = 0

    for epoch in range(epochs):
        model.train()
        train_loss, train_cm = 0, np.zeros((2, 2))
        for x, y in tqdm.tqdm(train_loader, desc=f"Train Epoch {epoch + 1}/{epochs}"):
            x, y = x.to(device), y.to(device).long()
            pred = model(x.float())
            loss = loss_fl(pred, y) + loss_jd(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_mPA, train_mDice, train_mIoU, train_recall, train_precision = metrice(y.cpu().numpy(),
                                                                                    pred.argmax(dim=1).cpu().numpy())
        with open(log_path, "a") as f:
            f.write(
                f"{epoch + 1},train,{train_loss / len(train_loader):.4f},{train_mPA:.4f},{train_mDice:.4f},{train_mIoU:.4f},{train_recall:.4f},{train_precision:.4f}\n"
            )

        tqdm.tqdm.write(
            f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, mPA: {train_mPA:.4f}, mDice: {train_mDice:.4f}, mIoU: {train_mIoU:.4f}")

        metrics["train_loss"].append(train_loss / len(train_loader))
        metrics["train_mPA"].append(train_mPA)
        metrics["train_mDice"].append(train_mDice)
        metrics["train_mIoU"].append(train_mIoU)
        metrics["train_recall"].append(train_recall)
        metrics["train_precision"].append(train_precision)

        test_loss = 0
        with torch.no_grad():
            for x, y in tqdm.tqdm(test_loader, desc=f"Test Epoch {epoch + 1}/{epochs}"):
                x, y = x.to(device), y.to(device).long()
                pred = model(x.float())
                test_loss += (loss_fl(pred, y) + loss_jd(pred, y)).item()
        test_mPA, test_mDice, test_mIoU, test_recall, test_precision = metrice(y.cpu().numpy(),
                                                                               pred.argmax(dim=1).cpu().numpy())

        with open(log_path, "a") as f:
            f.write(
                f"{epoch + 1},test,{test_loss / len(test_loader):.4f},{test_mPA:.4f},{test_mDice:.4f},{test_mIoU:.4f},{test_recall:.4f},{test_precision:.4f}\n"
            )

        tqdm.tqdm.write(
            f"Epoch {epoch + 1}/{epochs}, Test Loss: {test_loss:.4f}, mPA: {test_mPA:.4f}, mDice: {test_mDice:.4f}, mIoU: {test_mIoU:.4f}")

        metrics["test_loss"].append(test_loss / len(test_loader))
        metrics["test_mPA"].append(test_mPA)
        metrics["test_mDice"].append(test_mDice)
        metrics["test_mIoU"].append(test_mIoU)
        metrics["test_recall"].append(test_recall)
        metrics["test_precision"].append(test_precision)

        if test_mIoU > best_mIoU:
            best_mIoU = test_mIoU
            torch.save(model, best_model_path)

        if test_mPA > best_mPA:
            best_mPA = test_mPA

        if test_mDice > best_mDice:
            best_mDice = test_mDice

        if test_recall > best_recall:
            best_recall = test_recall

        if test_precision > best_precision:
            best_precision = test_precision

    plot_metrics(metrics, plot_path)

    with open(log_path, "w") as f:
        f.write("epoch,train_loss,test_loss,train_mPA,test_mPA,train_mDice,test_mDice,train_mIoU,test_mIoU,train_recall,test_recall,train_precision,test_precision\n")
        for i in range(epochs):
            f.write(f"{i + 1},{metrics['train_loss'][i]},{metrics['test_loss'][i]},"
                    f"{metrics['train_mPA'][i]},{metrics['test_mPA'][i]},"
                    f"{metrics['train_mDice'][i]},{metrics['test_mDice'][i]},"
                    f"{metrics['train_mIoU'][i]},{metrics['test_mIoU'][i]},"
                    f"{metrics['train_recall'][i]},{metrics['test_recall'][i]},"
                    f"{metrics['train_precision'][i]},{metrics['test_precision'][i]}\n")
        f.write(f"Best mPA: {best_mPA}, Best mDice: {best_mDice}, Best mIoU: {best_mIoU}, Best recall: {best_recall}, Best precision: {best_precision}\n")

    end_time = time.time()
    total_time = end_time - start_time

    with open(log_path, "a") as f:
        f.write(f"\nTotal Training Time (seconds): {total_time:.2f}\n")


def plot_metrics(metrics, plot_path):
    required_keys = ["train_loss", "test_loss", "train_mPA", "test_mPA",
                     "train_mDice", "test_mDice", "train_mIoU", "test_mIoU",
                     "train_recall", "test_recall", "train_precision", "test_precision"]
    for key in required_keys:
        if key not in metrics:
            raise ValueError(f"Missing key in metrics dictionary: {key}")

    num_epochs = len(metrics["train_loss"])
    if num_epochs == 0:
        raise ValueError("Metrics data is empty. Cannot generate plot.")

    epochs = range(1, num_epochs + 1)

    plt.figure(figsize=(12, 15))

    plt.subplot(3, 2, 1)
    plt.plot(epochs, metrics["train_loss"], "b", label="Train Loss", linewidth=2)
    plt.plot(epochs, metrics["test_loss"], "r", label="Test Loss", linewidth=2)
    plt.legend()
    plt.title("Loss")
    plt.xlabel("Epoch")

    plt.subplot(3, 2, 2)
    plt.plot(epochs, metrics["train_mPA"], "b", label="Train mPA", linewidth=2)
    plt.plot(epochs, metrics["test_mPA"], "r", label="Test mPA", linewidth=2)
    plt.legend()
    plt.title("Mean Pixel Accuracy (mPA)")
    plt.xlabel("Epoch")

    plt.subplot(3, 2, 3)
    plt.plot(epochs, metrics["train_mDice"], "b", label="Train mDice", linewidth=2)
    plt.plot(epochs, metrics["test_mDice"], "r", label="Test mDice", linewidth=2)
    plt.legend()
    plt.title("Mean Dice Coefficient (mDice)")
    plt.xlabel("Epoch")

    plt.subplot(3, 2, 4)
    plt.plot(epochs, metrics["train_mIoU"], "b", label="Train mIoU", linewidth=2)
    plt.plot(epochs, metrics["test_mIoU"], "r", label="Test mIoU", linewidth=2)
    plt.legend()
    plt.title("Mean Intersection over Union (mIoU)")
    plt.xlabel("Epoch")

    plt.subplot(3, 2, 5)
    plt.plot(epochs, metrics["train_recall"], "b", label="Train Recall", linewidth=2)
    plt.plot(epochs, metrics["test_recall"], "r", label="Test Recall", linewidth=2)
    plt.legend()
    plt.title("Recall")
    plt.xlabel("Epoch")

    plt.subplot(3, 2, 6)
    plt.plot(epochs, metrics["train_precision"], "b", label="Train Precision", linewidth=2)
    plt.plot(epochs, metrics["test_precision"], "r", label="Test Precision", linewidth=2)
    plt.legend()
    plt.title("Precision")
    plt.xlabel("Epoch")

    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print(f"Metrics plot saved to {plot_path}")


def metrice(y_true, y_pred):
    cm = confusion_matrix(y_true.flatten(), y_pred.flatten(), labels=[0, 1])

    mpa = np.nanmean(np.diag(cm) / np.maximum(cm.sum(axis=1), 1))

    iou = np.diag(cm) / (np.sum(cm, axis=1) + np.sum(cm, axis=0) - np.diag(cm) + 1e-7)
    miou = np.nanmean(iou)

    dice = 2 * np.diag(cm) / (np.sum(cm, axis=1) + np.sum(cm, axis=0) + 1e-7)
    mdice = np.nanmean(dice)

    tp = cm[1, 1]
    fn = cm[1, 0]
    fp = cm[0, 1]

    recall = tp / (tp + fn + 1e-7)
    precision = tp / (tp + fp + 1e-7)

    return mpa, mdice, miou, recall, precision


def generate_summary_report(results_dir, model_name):
    model_dir = os.path.join(results_dir, model_name)
    runs = [f"{i}th" for i in range(1, 6)]
    log_paths = [os.path.join(model_dir, run, "log.csv") for run in runs]

    best_metrics = []
    all_data = []

    for log_path in log_paths:
        if not os.path.exists(log_path):
            print(f"Warning: {log_path} not found. Skipping...")
            continue
        df = pd.read_csv(log_path)

        with open(log_path, "r") as f:
            last_lines = f.readlines()[-5:]

        best_line = next((line for line in last_lines if "Best mPA" in line), None)
        if best_line:
            values = [float(x.split(": ")[1]) for x in best_line.strip().split(", ")]
            if len(values) == 5:
                best_metrics.append(values)
            else:
                print(f"Warning: Incomplete best metrics found in {log_path}. Skipping...")

        epoch_data = df.iloc[:-3, 1:].to_numpy().astype(float)
        all_data.append(epoch_data)

    best_metrics = np.array(best_metrics)
    mean_metrics = np.mean(best_metrics, axis=0)
    std_metrics = np.std(best_metrics, axis=0)

    perf_path = os.path.join(model_dir, "performance.txt")
    with open(perf_path, "w") as f:
        f.write(f"Mean mPA: {mean_metrics[0]:.4f}, SD: {std_metrics[0]:.4f}\n")
        f.write(f"Mean mDice: {mean_metrics[1]:.4f}, SD: {std_metrics[1]:.4f}\n")
        f.write(f"Mean mIoU: {mean_metrics[2]:.4f}, SD: {std_metrics[2]:.4f}\n")
        f.write(f"Mean Recall: {mean_metrics[3]:.4f}, SD: {std_metrics[3]:.4f}\n")
        f.write(f"Mean Precision: {mean_metrics[4]:.4f}, SD: {std_metrics[4]:.4f}\n")

    print(f"Saved performance summary to {perf_path}")

    all_data = np.array(all_data)  # shape: (5, epochs, 8)
    mean_data = np.mean(all_data, axis=0)
    std_data = np.std(all_data, axis=0)

    summary_path = os.path.join(model_dir, "summary.csv")
    columns = ["train_loss", "test_loss", "train_mPA", "test_mPA", "train_mDice", "test_mDice",
               "train_mIoU", "test_mIoU", "train_recall", "test_recall", "train_precision", "test_precision"]
    df_summary = pd.DataFrame(np.hstack((mean_data, std_data)),
                              columns=[f"mean_{c}" for c in columns] + [f"std_{c}" for c in columns])
    df_summary.insert(0, "epoch", np.arange(1, len(df_summary) + 1))
    df_summary.to_csv(summary_path, index=False)

    print(f"Saved epoch-wise summary to {summary_path}")

    plot_path = os.path.join(model_dir, "indicators.png")
    plot_error_bands(df_summary, plot_path)



def plot_error_bands(df_summary, plot_path):
    required_cols = [
        "mean_train_loss", "std_train_loss", "mean_test_loss", "std_test_loss",
        "mean_train_mPA", "std_train_mPA", "mean_test_mPA", "std_test_mPA",
        "mean_train_mDice", "std_train_mDice", "mean_test_mDice", "std_test_mDice",
        "mean_train_mIoU", "std_train_mIoU", "mean_test_mIoU", "std_test_mIoU",
        "mean_train_recall", "std_train_recall", "mean_test_recall", "std_test_recall",
        "mean_train_precision", "std_train_precision", "mean_test_precision", "std_test_precision"
    ]

    for col in required_cols:
        if col not in df_summary:
            raise ValueError(f"Missing column in df_summary: {col}")

    epochs = df_summary["epoch"].to_numpy()
    if len(epochs) == 0:
        raise ValueError("df_summary is empty. Cannot generate plot.")

    plt.figure(figsize=(12, 10))

    def plot_subplot(pos, mean_train_col, std_train_col, mean_test_col, std_test_col, title):
        plt.subplot(3, 2, pos)

        mean_train = df_summary[mean_train_col].to_numpy()
        std_train = df_summary[std_train_col].to_numpy()
        mean_test = df_summary[mean_test_col].to_numpy()
        std_test = df_summary[std_test_col].to_numpy()

        plt.plot(epochs, mean_train, "b", label="Train", linewidth=2)
        plt.fill_between(epochs, mean_train - std_train, mean_train + std_train, color="b", alpha=0.3)

        plt.plot(epochs, mean_test, "r", label="Test", linewidth=2)
        plt.fill_between(epochs, mean_test - std_test, mean_test + std_test, color="r", alpha=0.3)

        plt.legend()
        plt.title(title)
        plt.xlabel("Epoch")

    plot_subplot(1, "mean_train_loss", "std_train_loss", "mean_test_loss", "std_test_loss", "Loss (Train vs Test)")
    plot_subplot(2, "mean_train_mPA", "std_train_mPA", "mean_test_mPA", "std_test_mPA", "mPA (Train vs Test)")
    plot_subplot(3, "mean_train_mDice", "std_train_mDice", "mean_test_mDice", "std_test_mDice", "mDice (Train vs Test)")
    plot_subplot(4, "mean_train_mIoU", "std_train_mIoU", "mean_test_mIoU", "std_test_mIoU", "mIoU (Train vs Test)")
    plot_subplot(5, "mean_train_recall", "std_train_recall", "mean_test_recall", "std_test_recall",
                 "Recall (Train vs Test)")
    plot_subplot(6, "mean_train_precision", "std_train_precision", "mean_test_precision", "std_test_precision",
                 "Precision (Train vs Test)")

    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved error band plot to {plot_path}")


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
