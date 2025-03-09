from __future__ import print_function
import os
import argparse
import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

from dataset import get_training_set
from utils import (
    setup_experiment, train_single_run, compute_summary, plot_error_bands,
    preprocess_images, set_seed
)

import warnings
warnings.simplefilter("ignore", UserWarning)

parser = argparse.ArgumentParser(description='pix2pix-pytorch-implementation')
parser.add_argument('--batch_size', type=int, default=4, help='training batch size')
parser.add_argument('--test_batch_size', type=int, default=1, help='testing batch size')
parser.add_argument('--direction', type=str, default='b2a', help='a2b or b2a')
parser.add_argument('--input_nc', type=int, default=3, help='input image channels')
parser.add_argument('--output_nc', type=int, default=3, help='output image channels')
parser.add_argument('--ngf', type=int, default=64, help='generator filters in first conv layer')
parser.add_argument('--ndf', type=int, default=64, help='discriminator filters in first conv layer')
parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count')
parser.add_argument('--niter', type=int, default=500, help='# of iter at starting learning rate')
parser.add_argument('--niter_decay', type=int, default=500, help='# of iter to linearly decay learning rate to zero')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy: lambda|step|plateau|cosine')
parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--lamb', type=int, default=10, help='weight on L1 term in objective')
opt = parser.parse_args()
print(opt)

set_seed(42)

cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image_dir = "<IMAGE_DIRECTORY>"
label_dir = "<LABEL_DIRECTORY>"
result_dir = "<OUTPUT_DIRECTORY>"

train_set = get_training_set(image_dir, label_dir, True)

print("Preprocessing validation data...")
preprocess_images(image_dir, label_dir, f"{result_dir}/temp")

data_sizes = [len(train_set)]

for data_size in data_sizes:
    experiment_dir = f"{result_dir}/{data_size}"

    if setup_experiment(data_size, experiment_dir):
        print(f"Skipping data size {data_size}, summary already exists.")
        continue

    for run_id in range(1, 6):
        run_dir = f"{experiment_dir}/{run_id}th"
        metrics_path = os.path.join(run_dir, "metrics.png")

        if os.path.exists(metrics_path):
            print(f"Skipping run {run_id} for data size {data_size}, metrics already exists.")
            continue

        sub_train_set = torch.utils.data.Subset(train_set, range(data_size))
        train_loader = DataLoader(dataset=sub_train_set, num_workers=opt.threads, batch_size=opt.batch_size, shuffle=True)

        train_single_run(run_dir, train_loader, f"{result_dir}/temp", device, opt, vis=False)

    indicator_path = os.path.join(experiment_dir, "indicators.png")
    summary_path = os.path.join(experiment_dir, "summary.csv")
    performance_path = os.path.join(experiment_dir, "performance.txt")

    if not (os.path.exists(summary_path) and os.path.exists(performance_path) and os.path.exists(indicator_path)):
        print(f"Missing summary or indicators for data size {data_size}. Recalculating...")
        compute_summary(experiment_dir, data_size)

        if os.path.exists(summary_path):
            plot_error_bands(experiment_dir, data_size)
        else:
            print(f"Error: summary.csv not generated correctly for data size {data_size}.")

print("Training complete! All datasets processed.")
