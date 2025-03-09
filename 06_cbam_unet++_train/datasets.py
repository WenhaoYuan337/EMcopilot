import os
import random
from glob import glob

import albumentations as A
import cv2
import torch
import torch.utils.data as data
from albumentations.pytorch import ToTensorV2


class MyDataset(data.Dataset):
    def __init__(self, image_dir, label_dir, transforms=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transforms = transforms

        self.mode = self.detect_mode()
        print(f"Dataset mode detected: {self.mode}")

        if self.mode == 'train':
            self.image_paths, self.label_paths = self.match_images_and_labels()
        else:
            self.image_paths = sorted(glob(os.path.join(image_dir, "*.png")))
            self.label_paths = sorted(glob(os.path.join(label_dir, "*.png")))

    def detect_mode(self):
        sample_images = glob(os.path.join(self.image_dir, "gen_*_*.png"))
        if sample_images:
            return 'train'
        else:
            return 'test'

    def match_images_and_labels(self):
        mask_files = sorted(glob(os.path.join(self.label_dir, "*.png")))
        image_paths = []
        label_paths = []

        for mask_path in mask_files:
            mask_filename = os.path.basename(mask_path)
            mask_id = mask_filename.split(".")[0]

            matched_images = glob(os.path.join(self.image_dir, f"gen_{mask_id}_*.png"))
            for image_path in matched_images:
                image_paths.append(image_path)
                label_paths.append(mask_path)

        return image_paths, label_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        label = cv2.imread(self.label_paths[idx], cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise ValueError(f"Failed to read image: {self.image_paths[idx]}")
        if label is None:
            raise ValueError(f"Failed to read label: {self.label_paths[idx]}")

        if self.transforms:
            augmented = self.transforms(image=image, mask=label)
            image = augmented["image"]
            label = augmented["mask"]

        label[label > 0] = 1

        return image, label


def get_fractional_data(image_dir, label_dir, batch_size, num_samples, transforms):
    dataset = MyDataset(image_dir, label_dir, transforms=transforms)

    if num_samples > len(dataset):
        raise ValueError(f"Requested {num_samples} samples, but only {len(dataset)} available.")

    indices = random.sample(range(len(dataset)), num_samples)
    subset = data.Subset(dataset, indices)

    loader = data.DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=4)
    return loader, len(subset)



def get_test_data(image_dir, label_dir, batch_size):
    test_transforms = A.Compose([
        A.Resize(512, 512),
        A.Normalize(),
        ToTensorV2(),
    ])

    dataset = MyDataset(image_dir, label_dir, transforms=test_transforms)

    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=False)

    return test_loader


def get_hybrid_data(image_dir1, label_dir1, image_dir2, label_dir2, batch_size, num_samples, num_samples_2=None,
                    hybrid_training=True):

    train_transforms = A.Compose([
        A.Resize(512, 512),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(p=0.5),
        A.GaussNoise(p=0.5),
        A.Normalize(),
        ToTensorV2(),
    ])

    test_transforms = A.Compose([
        A.Resize(512, 512),
        A.Normalize(),
        ToTensorV2(),
    ])

    dataset1 = MyDataset(image_dir1, label_dir1, transforms=train_transforms)
    dataset2 = MyDataset(image_dir2, label_dir2, transforms=test_transforms)

    if not hybrid_training:
        train_loader, train_size = get_fractional_data(image_dir1, label_dir1, batch_size, num_samples,
                                                       train_transforms)
        test_loader = get_test_data(image_dir2, label_dir2, batch_size)
        return train_loader, train_size, test_loader, len(dataset2)

    if num_samples > len(dataset1):
        raise ValueError(f"Requested {num_samples} samples, but only {len(dataset1)} available.")

    indices1 = random.sample(range(len(dataset1)), num_samples)
    subset1 = data.Subset(dataset1, indices1)

    total_lr = len(dataset2)

    if num_samples_2 is not None:
        if num_samples_2 > total_lr:
            raise ValueError(f"Requested {num_samples_2} samples, but only {total_lr} available")

        indices2 = random.sample(range(total_lr), num_samples_2)
        train_subset2 = data.Subset(dataset2, indices2)

        test_indices = sorted(set(range(total_lr)) - set(indices2))
        test_subset2 = data.Subset(dataset2, test_indices)

    else:
        train_size = int(total_lr * 0.8)
        test_size = total_lr - train_size
        train_subset2, test_subset2 = torch.utils.data.random_split(dataset2, [train_size, test_size])

    combined_dataset = data.ConcatDataset([subset1, train_subset2])
    train_loader = data.DataLoader(combined_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    test_loader = torch.utils.data.DataLoader(test_subset2, batch_size=batch_size, shuffle=False, num_workers=4,
                                              drop_last=False)

    return train_loader, len(combined_dataset), test_loader, len(test_subset2)





