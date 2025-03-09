import gc
import os
import random
import re
import time

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

os.environ['MPLBACKEND'] = 'Agg'
import matplotlib
matplotlib.use('Agg')
from PIL import Image
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchmetrics.image.fid import FrechetInceptionDistance
from skimage.metrics import structural_similarity as ssim
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.ssim import MultiScaleStructuralSimilarityIndexMeasure

from networks import define_G, define_D, GANLoss, SSIMLoss, get_scheduler, MSSSIMLoss, PerceptualLoss, update_learning_rate

import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
import os


import warnings
warnings.filterwarnings("ignore")


def setup_experiment(data_size, base_dir="result/fraction"):
    experiment_dir = os.path.join(base_dir, str(data_size))
    summary_path = os.path.join(experiment_dir, "summary.csv")
    performance_path = os.path.join(experiment_dir, "performance.txt")
    indicator_path = os.path.join(experiment_dir, "indicators.png")

    if not (os.path.exists(summary_path) and os.path.exists(performance_path) and os.path.exists(indicator_path)):
        return False

    return True



def train_single_run(run_dir, train_loader, val_dir, device, opt, vis=False):
    start_time = time.time()
    log_path = os.path.join(run_dir, "log.csv")
    metrics_path = os.path.join(run_dir, "metrics.png")

    if os.path.exists(metrics_path):
        print(f"Skipping existing training run: {run_dir}")
        return

    os.makedirs(run_dir, exist_ok=True)

    net_g = define_G(opt.input_nc, opt.output_nc, opt.ngf, 'batch', False, 'normal', 0.02, gpu_id=device).to(device)
    net_d = define_D(opt.input_nc + opt.output_nc, opt.ndf, 'basic', gpu_id=device).to(device)

    criterionGAN = GANLoss().to(device)
    criterionL1 = nn.L1Loss().to(device)
    criterionSSIM = SSIMLoss().to(device)
    criterionMSSSIM = MSSSIMLoss().to(device)
    criterionPerceptual = PerceptualLoss().to(device)

    optimizer_g = optim.Adam(net_g.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    optimizer_d = optim.Adam(net_d.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    net_g_scheduler = get_scheduler(optimizer_g, opt)
    net_d_scheduler = get_scheduler(optimizer_d, opt)

    best_ssim = 0.0
    best_msssim = 0.0
    best_fid = float("inf")
    best_lpips = 1
    best_G_GAN_loss = float("inf")
    best_G_L1_loss = float("inf")
    best_G_SSIM_loss = float("inf")
    best_G_MSSSIM_loss = float("inf")
    best_G_Perceptual_loss = float("inf")
    best_D_real_loss = float("inf")
    best_D_fake_loss = float("inf")


    with open(log_path, "w") as f:
        f.write("epoch,D_real_loss,D_fake_loss,G_GAN_loss,G_L1_loss,SSIM,MSSSIM,FID,LPIPS\n")

    metrics = {
        "epoch": [],
        "D_real_loss": [],
        "D_fake_loss": [],
        "G_GAN_loss": [],
        "G_L1_loss": [],
        "SSIM": [],
        "MSSSIM": [],
        "FID": [],
        "LPIPS": [],
    }

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        total_D_real_loss, total_D_fake_loss, total_G_GAN_loss, total_G_L1_loss = 0, 0, 0, 0
        print(f"\n")
        for batch in tqdm(train_loader, desc=f"Training {run_dir}, Epoch {epoch}"):
            real_a, real_b = batch[0].to(device), batch[1].to(device)
            fake_b = net_g(real_a)

            optimizer_d.zero_grad()

            fake_ab = torch.cat((real_a, fake_b), 1)
            pred_fake = net_d(fake_ab.detach())
            D_fake_loss = criterionGAN(pred_fake, False)

            real_ab = torch.cat((real_a, real_b), 1)
            pred_real = net_d(real_ab)
            D_real_loss = criterionGAN(pred_real, True)

            D_loss = (D_real_loss + D_fake_loss) * 0.5
            D_loss.backward()
            optimizer_d.step()

            optimizer_g.zero_grad()
            fake_ab = torch.cat((real_a, fake_b), 1)
            pred_fake = net_d(fake_ab)
            G_GAN_loss = criterionGAN(pred_fake, True)
            G_L1_loss = criterionL1(fake_b, real_b)
            lambda_L1 = 50

            G_loss = G_GAN_loss + lambda_L1 * G_L1_loss
            G_loss.backward()
            optimizer_g.step()

            total_D_real_loss += D_real_loss.item()
            total_D_fake_loss += D_fake_loss.item()
            total_G_GAN_loss += G_GAN_loss.item()
            total_G_L1_loss += G_L1_loss.item()

        torch.cuda.empty_cache()
        gc.collect()

        # checkpoint_dir = os.path.join(run_dir, "checkpoint")
        # os.makedirs(checkpoint_dir, exist_ok=True)
        # torch.save(net_g, os.path.join(checkpoint_dir, f"net_g_{epoch}.pth"))

        # 计算 SSIM
        current_ssim, current_msssim, current_fid, current_lpips = validate_model(net_g, val_dir, device, epoch,
                                                                                  vis_enabled=vis)

        tqdm.write(f"Validation - SSIM: {current_ssim:.4f}, MSSSIM: {current_msssim:.4f}, FID: {current_fid:.4f}, LPIPS: {current_lpips:.4f}")

        if total_G_GAN_loss < G_GAN_loss:
            best_G_GAN_loss = total_G_GAN_loss

        if total_G_L1_loss < G_L1_loss:
            best_G_L1_loss = total_G_L1_loss

        if total_D_real_loss < best_D_real_loss:
            best_D_real_loss = total_D_real_loss

        if total_D_fake_loss < best_D_fake_loss:
            best_D_fake_loss = total_D_fake_loss

        if current_ssim > best_ssim:
            best_ssim = current_ssim
            torch.save(net_g, os.path.join(run_dir, "best_ssim_model.pth"))
            tqdm.write(f"===> New best SSIM: {best_ssim:.4f}")

        if current_msssim > best_msssim:
            best_msssim = current_msssim
            torch.save(net_g, os.path.join(run_dir, "best_msssim_model.pth"))
            tqdm.write(f"===> New best MSSSIM: {best_msssim:.4f}")

        if current_fid < best_fid:
            best_fid = current_fid
            torch.save(net_g, os.path.join(run_dir, "best_fid_model.pth"))
            tqdm.write(f"===> New best FID: {best_fid:.4f}")

        if current_lpips < best_lpips:
            best_lpips = current_lpips
            torch.save(net_g, os.path.join(run_dir, "best_lpips_model.pth"))
            tqdm.write(f"===> New best LPIPS: {best_lpips:.4f}")

        if epoch == opt.niter + opt.niter_decay:
            torch.save(net_g, os.path.join(run_dir, "last_epoch_model.pth"))
            tqdm.write(f"===> Last epoch model saved")

        metrics["epoch"].append(epoch)
        metrics["D_real_loss"].append(total_D_real_loss)
        metrics["D_fake_loss"].append(total_D_fake_loss)
        metrics["G_GAN_loss"].append(total_G_GAN_loss)
        metrics["G_L1_loss"].append(total_G_L1_loss)
        metrics["SSIM"].append(current_ssim)
        metrics["MSSSIM"].append(current_msssim)
        metrics["FID"].append(current_fid)
        metrics["LPIPS"].append(current_lpips)


        update_learning_rate(net_g_scheduler, optimizer_g)
        update_learning_rate(net_d_scheduler, optimizer_d)

    with open(log_path, "w") as f:
        f.write("epoch,D_real_loss,D_fake_loss,G_GAN_loss,G_L1_loss,SSIM,MSSSIM,FID,LPIPS\n")
        for i in range(len(metrics["epoch"])):
            f.write(f"{metrics['epoch'][i]},{metrics['D_real_loss'][i]},{metrics['D_fake_loss'][i]},"
                    f"{metrics['G_GAN_loss'][i]},{metrics['G_L1_loss'][i]},{metrics['SSIM'][i]},"
                    f"{metrics['MSSSIM'][i]},{metrics['FID'][i]},{metrics['LPIPS'][i]:.4f}\n")

        f.write(f"Best D_real_loss: {best_D_real_loss}, "
                f"Best D_fake_loss: {best_D_fake_loss}, Best G_GAN_loss: {best_G_GAN_loss}, "
                f"Best G_L1_loss: {best_G_L1_loss}, Best SSIM: {best_ssim}, "
                f"Best FID: {best_msssim}, Best LPIPS: {best_fid}, Best Z-score: {best_lpips:.4f}\n")

    plot_metrics(log_path, metrics_path)

    del net_g, net_d, optimizer_g, optimizer_d
    gc.collect()
    torch.cuda.empty_cache()

    end_time = time.time()
    total_time = end_time - start_time

    with open(log_path, "a") as f:
        f.write(f"\nTotal Training Time (seconds): {total_time:.2f}\n")




def compute_summary(result_dir, data_size, num_runs=5):
    print(f"Computing summary for {result_dir} (data size: {data_size})...")

    log_paths = [os.path.join(result_dir, f"{i}th/log.csv") for i in range(1, num_runs + 1)]
    epoch_data = []
    best_values = []

    for log_path in log_paths:
        if not os.path.exists(log_path):
            print(f"Error: Log file {log_path} not found!")
            return

        df = pd.read_csv(log_path)

        df = df.apply(pd.to_numeric, errors="coerce")
        df = df.dropna()

        if df.empty:
            print(f"Error: Log file {log_path} contains no valid numerical data!")
            return

        df = df[df["epoch"].apply(lambda x: isinstance(x, (int, float)))]

        if "FID" not in df.columns or "LPIPS" not in df.columns:
            print(f"Error: Missing FID or LPIPS columns in {log_path}")
            return

        epoch_data.append(df)

        with open(log_path, "r") as f:
            lines = f.readlines()
            for line in reversed(lines):
                if "Best D_real_loss" in line:
                    try:
                        parts = line.strip().split(", ")
                        best_D_real_loss = float(parts[0].split(": ")[1])
                        best_D_fake_loss = float(parts[1].split(": ")[1])
                        best_G_GAN_loss = float(parts[2].split(": ")[1])
                        best_G_L1_loss = float(parts[3].split(": ")[1])
                        best_SSIM = float(parts[4].split(": ")[1])
                        best_MSSSIM = float(parts[5].split(": ")[1])
                        best_FID = float(parts[6].split(": ")[1])
                        best_LPIPS = float(parts[7].split(": ")[1])
                        best_values.append([
                            best_D_real_loss, best_D_fake_loss, best_G_GAN_loss, best_G_L1_loss,
                            best_SSIM, best_MSSSIM, best_FID, best_LPIPS
                        ])
                    except ValueError:
                        print(f"Error: Failed to parse best values in {log_path}")
                    break

    if not epoch_data:
        print(f"Error: No valid log data found for {result_dir}!")
        return

    summary_path = os.path.join(result_dir, "summary.csv")
    print(f"Saving summary.csv to {summary_path}")

    min_len = min(len(df) for df in epoch_data)
    epoch_data = [df.iloc[:min_len].to_numpy(dtype=float) for df in epoch_data]
    epoch_data = np.array(epoch_data, dtype=float)

    mean_epoch_data = np.mean(epoch_data, axis=0)
    std_epoch_data = np.std(epoch_data, axis=0)

    columns = [
        "epoch", "D_real_loss", "D_fake_loss", "G_GAN_loss", "G_L1_loss", "SSIM", "MSSSIM", "FID", "LPIPS"
    ]

    df_summary = pd.DataFrame(
        np.hstack((mean_epoch_data, std_epoch_data)),
        columns=[f"mean_{c}" for c in columns] + [f"std_{c}" for c in columns]
    )

    df_summary.to_csv(summary_path, index=False)

    if not os.path.exists(summary_path) or os.path.getsize(summary_path) == 0:
        print(f"Error: summary.csv not created successfully at {summary_path}!")
        return

    if len(best_values) == 0:
        print(f"Error: No best values found for {result_dir}. Skipping performance calculation.")
        return

    best_values = np.array(best_values, dtype=float)
    if best_values.shape[1] < 8:
        print("Error: Best values array does not contain all required metrics. Check log files.")
        return
    mean_best = np.mean(best_values, axis=0)
    std_best = np.std(best_values, axis=0)

    performance_path = os.path.join(result_dir, "performance.txt")
    with open(performance_path, "w") as f:
        f.write(f"Mean D_real_loss: {mean_best[0]:.4f}, Std: {std_best[0]:.4f}\n")
        f.write(f"Mean D_fake_loss: {mean_best[1]:.4f}, Std: {std_best[1]:.4f}\n")
        f.write(f"Mean G_GAN_loss: {mean_best[2]:.4f}, Std: {std_best[2]:.4f}\n")
        f.write(f"Mean G_L1_loss: {mean_best[3]:.4f}, Std: {std_best[3]:.4f}\n")
        f.write(f"Mean SSIM: {mean_best[4]:.4f}, Std: {std_best[4]:.4f}\n")
        f.write(f"Mean MSSSIM: {mean_best[5]:.4f}, Std: {std_best[5]:.4f}\n")
        f.write(f"Mean FID: {mean_best[6]:.4f}, Std: {std_best[6]:.4f}\n")
        f.write(f"Mean LPIPS: {mean_best[7]:.4f}, Std: {std_best[7]:.4f}\n")

    print(f"Saved summary to {summary_path}")
    print(f"Saved performance report to {performance_path}")


def plot_error_bands(result_dir, data_size):
    summary_path = os.path.join(result_dir, "summary.csv")
    if not os.path.exists(summary_path):
        print(f"Warning: {summary_path} not found. Skipping plot_error_bands.")
        return

    df = pd.read_csv(summary_path)

    epochs = df["mean_epoch"].to_numpy()
    mean_D_real_loss = df["mean_D_real_loss"].to_numpy()
    std_D_real_loss = df["std_D_real_loss"].to_numpy()
    mean_D_fake_loss = df["mean_D_fake_loss"].to_numpy()
    std_D_fake_loss = df["std_D_fake_loss"].to_numpy()
    mean_G_GAN_loss = df["mean_G_GAN_loss"].to_numpy()
    std_G_GAN_loss = df["std_G_GAN_loss"].to_numpy()
    mean_G_L1_loss = df["mean_G_L1_loss"].to_numpy()
    std_G_L1_loss = df["std_G_L1_loss"].to_numpy()
    mean_SSIM = df["mean_SSIM"].to_numpy()
    std_SSIM = df["std_SSIM"].to_numpy()
    mean_MSSSIM = df["mean_MSSSIM"].to_numpy()
    std_MSSSIM = df["std_MSSSIM"].to_numpy()
    mean_FID = df["mean_FID"].to_numpy()
    std_FID = df["std_FID"].to_numpy()
    mean_LPIPS = df["mean_LPIPS"].to_numpy()
    std_LPIPS = df["std_LPIPS"].to_numpy()

    fig, axes = plt.subplots(3, 2, figsize=(12, 15), sharex=True)

    axes[0, 0].plot(epochs, mean_D_real_loss, "r", label="D_real_loss")
    axes[0, 0].fill_between(epochs, mean_D_real_loss - std_D_real_loss, mean_D_real_loss + std_D_real_loss, color="r", alpha=0.3)

    axes[0, 0].plot(epochs, mean_D_fake_loss, "orange", label="D_fake_loss")
    axes[0, 0].fill_between(epochs, mean_D_fake_loss - std_D_fake_loss, mean_D_fake_loss + std_D_fake_loss, color="orange", alpha=0.3)

    axes[0, 0].set_title("D Loss per Epoch", fontsize=14)
    axes[0, 0].set_ylabel("Loss", fontsize=12)
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    axes[0, 1].plot(epochs, mean_G_GAN_loss, "b", label="G_GAN_loss")
    axes[0, 1].fill_between(epochs, mean_G_GAN_loss - std_G_GAN_loss, mean_G_GAN_loss + std_G_GAN_loss, color="b", alpha=0.3)

    axes[0, 1].plot(epochs, mean_G_L1_loss, "g", label="G_L1_loss")
    axes[0, 1].fill_between(epochs, mean_G_L1_loss - std_G_L1_loss, mean_G_L1_loss + std_G_L1_loss, color="g", alpha=0.3)

    axes[0, 1].set_title("G Loss per Epoch", fontsize=14)
    axes[0, 1].set_ylabel("Loss", fontsize=12)
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    axes[1, 0].plot(epochs, mean_SSIM, "r", label="SSIM")
    axes[1, 0].fill_between(epochs, mean_SSIM - std_SSIM, mean_SSIM + std_SSIM, color="r", alpha=0.3)

    axes[1, 0].set_title("SSIM per Epoch", fontsize=14)
    axes[1, 0].set_ylabel("SSIM Score", fontsize=12)
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    axes[1, 1].plot(epochs, mean_MSSSIM, "k", label="MSSSIM")
    axes[1, 1].fill_between(epochs, mean_MSSSIM - std_MSSSIM, mean_MSSSIM + std_MSSSIM, color="gray", alpha=0.3)

    axes[1, 1].set_title("MSSSIM per Epoch", fontsize=14)
    axes[1, 1].set_ylabel("MSSSIM", fontsize=12)
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    axes[2, 0].plot(epochs, mean_FID, "c", label="FID")
    axes[2, 0].fill_between(epochs, mean_FID - std_FID, mean_FID + std_FID, color="c", alpha=0.3)

    axes[2, 0].set_title("FID per Epoch", fontsize=14)
    axes[2, 0].set_ylabel("FID Score", fontsize=12)
    axes[2, 0].legend()
    axes[2, 0].grid(True)

    axes[2, 1].plot(epochs, mean_LPIPS, "m", label="LPIPS")
    axes[2, 1].fill_between(epochs, mean_LPIPS - std_LPIPS, mean_LPIPS + std_LPIPS, color="m", alpha=0.3)

    axes[2, 1].set_title("LPIPS per Epoch", fontsize=14)
    axes[2, 1].set_ylabel("LPIPS Score", fontsize=12)
    axes[2, 1].legend()
    axes[2, 1].grid(True)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.15, wspace=0.15)
    plt.savefig(os.path.join(result_dir, "indicators.png"))
    plt.close()

    print(f"Saved error band plot to {os.path.join(result_dir, 'indicators.png')}")


def plot_metrics(log_path, save_path):
    if not os.path.exists(log_path):
        print(f"Error: {log_path} not found!")
        return

    df = pd.read_csv(log_path)

    df = df[~df["epoch"].astype(str).str.contains("Best", na=False)]

    df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce").dropna().astype(int)

    required_columns = ["epoch", "D_real_loss", "D_fake_loss", "G_GAN_loss", "G_L1_loss", "SSIM", "MSSSIM", "FID", "LPIPS"]
    if not all(col in df.columns for col in required_columns):
        print(f"Error: Missing required columns in {log_path}")
        return

    for col in required_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").dropna()

    epochs = df["epoch"].to_numpy()
    D_real_loss = df["D_real_loss"].to_numpy()
    D_fake_loss = df["D_fake_loss"].to_numpy()
    G_GAN_loss = df["G_GAN_loss"].to_numpy()
    G_L1_loss = df["G_L1_loss"].to_numpy()
    SSIM = df["SSIM"].to_numpy()
    MSSSIM = df["MSSSIM"].to_numpy()
    FID = df["FID"].to_numpy()
    LPIPS = df["LPIPS"].to_numpy()

    if len(epochs) == 0:
        print(f"Error: No valid epoch data found in {log_path}")
        return

    fig, axes = plt.subplots(3, 2, figsize=(12, 15), sharex=True)

    axes[0, 0].plot(epochs, D_real_loss, marker="o", color="red", label="D_real_loss")
    axes[0, 0].plot(epochs, D_fake_loss, marker="o", color="green", label="D_fake_loss")
    axes[0, 0].set_title("D Loss per Epoch", fontsize=14)
    axes[0, 0].set_ylabel("Loss", fontsize=12)
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    axes[0, 1].plot(epochs, G_GAN_loss, marker="o", color="blue", label="G_GAN_loss")
    axes[0, 1].plot(epochs, G_L1_loss, marker="o", color="orange", label="G_L1_loss")
    axes[0, 1].set_title("G Loss per Epoch", fontsize=14)
    axes[0, 1].set_ylabel("Loss", fontsize=12)
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    axes[1, 0].plot(epochs, SSIM, marker="o", color="purple", label="SSIM")
    axes[1, 0].set_title("SSIM per Epoch", fontsize=14)
    axes[1, 0].set_ylabel("SSIM Score", fontsize=12)
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    axes[1, 1].plot(epochs, MSSSIM, marker="o", color="cyan", label="MSSSIM")
    axes[1, 1].set_title("MSSSIM per Epoch", fontsize=14)
    axes[1, 1].set_ylabel("MSSSIM Score", fontsize=12)
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    axes[2, 0].plot(epochs, FID, marker="o", color="brown", label="FID")
    axes[2, 0].set_title("FID per Epoch", fontsize=14)
    axes[2, 0].set_xlabel("Epoch", fontsize=12)
    axes[2, 0].set_ylabel("FID Score", fontsize=12)
    axes[2, 0].legend()
    axes[2, 0].grid(True)

    axes[2, 1].plot(epochs, LPIPS, marker="o", color="magenta", label="LPIPS")
    axes[2, 1].set_title("LPIPS per Epoch", fontsize=14)
    axes[2, 1].set_xlabel("Epoch", fontsize=12)
    axes[2, 1].set_ylabel("LPIPS", fontsize=12)
    axes[2, 1].legend()
    axes[2, 1].grid(True)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.15, wspace=0.15)

    plt.savefig(save_path)
    plt.close()

    print(f"Saved training metrics plot to {save_path}")


def validate_model(net_g, temp_dir, device, epoch, vis_enabled=False):
    net_g.to(device)
    net_g.eval()

    total_ssim = 0.0
    total_msssim = 0.0
    total_lpips = 0.0
    label_files = sorted(
        [f for f in os.listdir(temp_dir) if f.startswith("label_")],
        key=lambda x: int(re.search(r'\d+', x).group())
    )
    num_images = len(label_files)

    if num_images == 0:
        print("Warning: No images found for validation.")
        return 0.0, 0.0, 0.0

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化到 [-1,1]
    ])

    if vis_enabled:
        index_to_select = 8 if len(label_files) > 8 else len(label_files) - 1
        selected_label_name = label_files[index_to_select]

        vis_dir = os.path.join(temp_dir, "vis")
        os.makedirs(vis_dir, exist_ok=True)

        selected_label_path = os.path.join(temp_dir, selected_label_name)
        label_np = np.load(selected_label_path)

        if len(label_np.shape) == 2:
            label_np = np.stack([label_np] * 3, axis=-1)

        label_tensor = transform(Image.fromarray(label_np)).unsqueeze(0).to(device)

        with torch.no_grad():
            generated_tensor = net_g(label_tensor)

        generated_tensor = (generated_tensor * 0.5 + 0.5).clamp(0, 1)
        save_image(generated_tensor, os.path.join(vis_dir, f"epoch_{epoch}_vis_generated.png"))

        print(f"Visualization saved in {vis_dir} for Epoch {epoch}: epoch_{epoch}_vis_generated.png")


    fid = FrechetInceptionDistance(normalize=True).to(device)
    lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg').to(device)

    for img_name in label_files:
        img_basename = img_name.replace("label_", "").replace(".npy", "")
        real_np = np.load(os.path.join(temp_dir, f"real_{img_basename}.npy"))
        label_np = np.load(os.path.join(temp_dir, f"label_{img_basename}.npy"))

        if len(real_np.shape) == 2:
            real_np = np.stack([real_np] * 3, axis=-1)
        if len(label_np.shape) == 2:
            label_np = np.stack([label_np] * 3, axis=-1)

        label_tensor = transform(Image.fromarray(label_np)).unsqueeze(0).to(device)

        with torch.no_grad():
            generated_tensor = net_g(label_tensor)

        generated_tensor = (generated_tensor * 0.5 + 0.5).clamp(0, 1)
        real_tensor = transform(Image.fromarray(real_np)).unsqueeze(0).to(device)
        generated_np = generated_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255
        generated_np = generated_np.astype(np.uint8)

        ssim_score = ssim(real_np, generated_np, channel_axis=-1)
        total_ssim += ssim_score

        real_tensor = transform(Image.fromarray(real_np)).unsqueeze(0).to(device)
        fid.update(real_tensor, real=True)
        fid.update(generated_tensor, real=False)

        total_lpips += lpips(real_tensor, generated_tensor).item()

        generated_tensor = generated_tensor.clamp(0, 1)
        real_tensor = real_tensor.clamp(0, 1)

        if len(generated_tensor.shape) == 3:
            generated_tensor = generated_tensor.permute(2, 0, 1).unsqueeze(0)
        if len(real_tensor.shape) == 3:
            real_tensor = real_tensor.permute(2, 0, 1).unsqueeze(0)

        msssim_metric = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        msssim_score = msssim_metric(generated_tensor, real_tensor)

        total_msssim += msssim_score.item()

    net_g.train()
    avg_ssim = total_ssim / num_images
    avg_msssim = total_msssim / num_images
    fid_score = fid.compute().item() / num_images
    avg_lpips = total_lpips / num_images

    fid.reset()
    lpips.reset()
    del net_g, fid, lpips
    torch.cuda.empty_cache()

    return avg_ssim, avg_msssim, fid_score, avg_lpips




def preprocess_images(image_dir, label_dir, save_dir, target_size=(512, 512)):
    os.makedirs(save_dir, exist_ok=True)

    for img_name in os.listdir(image_dir):
        img_path = os.path.join(image_dir, img_name)
        label_path = os.path.join(label_dir, img_name)

        real_img = np.array(Image.open(img_path).convert("RGB").resize(target_size, Image.BICUBIC))
        label_img = np.array(Image.open(label_path).convert("RGB").resize(target_size, Image.BICUBIC))

        np.save(os.path.join(save_dir, f"real_{img_name}.npy"), real_img)
        np.save(os.path.join(save_dir, f"label_{img_name}.npy"), label_img)


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    img = img.resize((512, 512), Image.BICUBIC)
    return img


def save_img(image_tensor, filename):
    image_numpy = image_tensor.float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    image_numpy = image_numpy.clip(0, 255)
    image_numpy = image_numpy.astype(np.uint8)
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(filename)
    print("Image saved as {}".format(filename))


def set_seed(seed=42):
    """
    Set random seeds to ensure reproducibility.

    Args:
        seed (int, optional): The random seed value. Defaults to 42.
    """
    print(f"Setting random seed: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # For GPU
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

