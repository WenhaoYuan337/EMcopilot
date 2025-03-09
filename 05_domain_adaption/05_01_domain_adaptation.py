import os

import cv2
import numpy as np
from PIL import Image

input_folder = "<INPUT_DIRECTORY>"
output_folder = "<OUTPUT_DIRECTORY>"

os.makedirs(output_folder, exist_ok=True)

sigmoid_params = [(2, 0.3), (5, 0.3)]

noise_strategies = [
    ("shot_150", lambda img: add_shot_noise(img, intensity=150)),
    ("shot_250", lambda img: add_shot_noise(img, intensity=250)),
    ("gaussian_0.03", lambda img: add_gaussian_noise(img, mean=0, std=0.03)),
    ("gaussian_0.05", lambda img: add_gaussian_noise(img, mean=0, std=0.05)),
    ("shot_250_gaussian_0.03", lambda img: add_gaussian_noise(add_shot_noise(img, intensity=250), mean=0, std=0.03)),
    ("shot_250_gaussian_0.03_scan", lambda img: add_scan_noise_smoothed(add_gaussian_noise(add_shot_noise(img, intensity=250), mean=0, std=0.03), sigma_jitter=2, freq=0.001))
]

def add_shot_noise(image, intensity=100):
    scale = intensity
    noisy_image = np.random.poisson(image * scale) / scale
    return np.clip(noisy_image, 0, 1)

def add_scan_noise_smoothed(image, sigma_jitter=2, freq=0.001):
    noisy_image = image.copy()
    height, width = image.shape

    for i in range(height):
        random_factor = 0.5 + np.random.rand()
        jitter_x = np.random.normal(0, sigma_jitter) * np.sin(2 * np.pi * freq * i) * random_factor
        jitter_x = int(round(jitter_x))

        if abs(jitter_x) > 0:
            row = noisy_image[i].copy()
            row = np.roll(row, jitter_x)
            row[:abs(jitter_x)] = np.mean(row)
            noisy_image[i] = row

    return np.clip(noisy_image, 0, 1)

def add_gaussian_noise(image, mean=0, std=0.05):
    gaussian_noise = np.random.normal(mean, std, image.shape)
    noisy_image = image + gaussian_noise
    return np.clip(noisy_image, 0, 1)

def sigmoid_correction(image, k, i0):
    np_img = np.array(image, dtype=np.float32) / 255.0
    sigmoid = 1 / (1 + np.exp(-k * (np_img - i0)))
    sigmoid = (sigmoid - 1 / (1 + np.exp(k * i0))) / (1 - 1 / (1 + np.exp(k * i0)))
    adjusted_img = sigmoid * 255.0
    adjusted_img = np.clip(adjusted_img, 0, 255).astype(np.uint8)
    return Image.fromarray(adjusted_img)

for file_name in sorted(os.listdir(input_folder)):
    if not file_name.lower().endswith(".png"):
        continue

    input_path = os.path.join(input_folder, file_name)
    image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Failed to load {file_name}")
        continue

    image = image.astype(np.float32) / 255.0

    for strategy_name, noise_func in noise_strategies:
        noisy_image = noise_func(image)
        noisy_image = (noisy_image * 255).astype(np.uint8)

        for k, i0 in sigmoid_params:
            noisy_pil = Image.fromarray(noisy_image)
            adjusted_image = sigmoid_correction(noisy_pil, k, i0)

            output_filename = f"{os.path.splitext(file_name)[0]}_{strategy_name}_sigmoid_k{k}_i0{i0}.png"
            output_path = os.path.join(output_folder, output_filename)
            adjusted_image.save(output_path)
            print(f"Processed {output_filename}")

        cv2.imwrite(
            os.path.join(output_folder, f"{os.path.splitext(file_name)[0]}_{strategy_name}_original.png"), noisy_image
        )

print(f"Processing complete. Files saved in {output_folder}.")
