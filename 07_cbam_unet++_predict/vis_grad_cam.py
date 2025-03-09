import os

import albumentations as A
import cv2
import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import SemanticSegmentationTarget

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_aug = A.Compose([
    A.Resize(512, 512),
    A.Normalize(),
])

model = torch.load('<MODEL_PATH>', map_location=DEVICE)
model.to(DEVICE)
model.eval()

block_names = list(model.unet.decoder.blocks.keys())
print("Decoder blocks:", block_names)

last_block_name = block_names[-1]
target_layer = model.unet.decoder.blocks[last_block_name].conv2[0]
print(f"Using target layer: {target_layer}")

cam = GradCAM(model=model, target_layers=[target_layer])

save_path = "segmentation_results/ML/attention"
raw_data_path = os.path.join(save_path, "raw_cam_data")
csv_data_path = os.path.join(save_path, "raw_cam_data_csv")
os.makedirs(save_path, exist_ok=True)
os.makedirs(raw_data_path, exist_ok=True)
os.makedirs(csv_data_path, exist_ok=True)

test_images_path = "<IMAGE_PATH>"
image_filenames = os.listdir(test_images_path)

for img_name in image_filenames:
    ori_image = cv2.imread(os.path.join(test_images_path, img_name))
    ori_image_rgb = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB) / 255.0
    ori_image_rgb = cv2.resize(ori_image_rgb, (512, 512))

    image = test_aug(image=ori_image)['image']
    image = np.expand_dims(np.transpose(image, axes=[2, 0, 1]), axis=0)
    image_tensor = torch.from_numpy(image).to(DEVICE).float()

    output = model(image_tensor)
    pred_mask = output.argmax(1).squeeze().cpu().numpy()

    target_category = 1

    mask = np.zeros(image_tensor.shape[2:], dtype=np.uint8)
    mask[pred_mask == target_category] = 1
    targets = [SemanticSegmentationTarget(target_category, mask)]
    grayscale_cam = cam(input_tensor=image_tensor, targets=targets)[0]
    heatmap = show_cam_on_image(ori_image_rgb, grayscale_cam, use_rgb=True)

    cv2.imwrite(os.path.join(save_path, f"grad_cam_{img_name}"), cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR))

print("Grad-CAM 计算完成，所有结果已保存。")
