import glob
import os
import torch
from datasets import get_hybrid_data
from model import AttentionUNetPlusPlus
from util import train_model, generate_summary_report, set_seed

if __name__ == "__main__":
    EPOCHS = 20
    BATCH_SIZE = 8
    DEVICE = torch.device("cuda")
    RESULTS_DIR = "<RESULT_DIRECTORY>"
    hybrid_training = False

    image_dir1 = "<IMAGE_1_DIRECTORY>"
    label_dir1 = "<LABEL_1_DIRECTORY>"
    image_dir2 = "<IMAGE_2_DIRECTORY>"
    label_dir2 = "<LABEL_2_DIRECTORY>"

    total_samples = len(glob.glob(os.path.join(image_dir1, "*.png")))
    model_name = "CBAM_UNet++"
    model = AttentionUNetPlusPlus(encoder_name="resnet34", classes=2, attention_type="CBAM").to(DEVICE)

    sample_sizes = [5940]
    sample_sizes_2 = [0, 1, 3, 5]

    if not sample_sizes_2:
        sample_sizes_2 = [None]

    for num_samples in sample_sizes:
        for num_samples_2 in sample_sizes_2:
            model = AttentionUNetPlusPlus(encoder_name="resnet34", classes=2, attention_type="CBAM").to(DEVICE)
            for i in range(1, 6):
                run_results_dir = os.path.join(RESULTS_DIR,
                                               f"{num_samples}_{num_samples_2 if num_samples_2 is not None else 'default'}",
                                               f"{i}th")
                os.makedirs(run_results_dir, exist_ok=True)
                set_seed(42 + i)
                train_loader, train_size, test_loader, test_size = get_hybrid_data(
                    image_dir1, label_dir1, image_dir2, label_dir2, BATCH_SIZE, num_samples, num_samples_2,
                    hybrid_training
                )
                train_model(model, model_name, train_loader, test_loader, DEVICE, EPOCHS, run_results_dir)

        summary_model_name = f"{num_samples}_{num_samples_2 if num_samples_2 is not None else 'default'}"
        summary_path = os.path.join(RESULTS_DIR, summary_model_name, "summary.csv")
        performance_path = os.path.join(RESULTS_DIR, summary_model_name, "performance.txt")
        indicators_path = os.path.join(RESULTS_DIR, summary_model_name, "indicators.png")

        if not (os.path.exists(summary_path) and os.path.exists(performance_path) and os.path.exists(indicators_path)):
            generate_summary_report(RESULTS_DIR, summary_model_name)
