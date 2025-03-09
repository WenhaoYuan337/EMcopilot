from __future__ import print_function

import os

import torch
import torchvision.transforms as transforms

from utils import is_image_file, load_img, save_img

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = "<MODEL_DIRECTORY>"
net_g = torch.load(model_path, map_location=device)

image_dir = "<INPUT_DIRECTORY>"
output_dir = "<OUTPUT_DIRECTORY>"

os.makedirs(output_dir, exist_ok=True)

image_filenames = [x for x in os.listdir(image_dir) if is_image_file(x) and x.endswith(".png")]
image_filenames.sort(key=lambda x: int(os.path.splitext(x)[0]))

transform_list = [
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
]
transform = transforms.Compose(transform_list)

for image_name in image_filenames:
    img = load_img(os.path.join(image_dir, image_name))
    img = transform(img)
    input_img = img.unsqueeze(0).to(device)

    with torch.no_grad():
        out = net_g(input_img)

    out_img = out.detach().squeeze(0).cpu()
    seq_number = os.path.splitext(image_name)[0]
    new_filename = f"gen_{seq_number}.png"
    save_img(out_img, os.path.join(output_dir, new_filename))
    print(f"Processed: {image_name} -> {new_filename}")

print("Processing complete!")
