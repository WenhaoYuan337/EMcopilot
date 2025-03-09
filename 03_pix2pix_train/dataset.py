from os import listdir
from os.path import join

import albumentations as A
import cv2
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image

from utils import is_image_file


class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, label_dir, augment=False):
        super(DatasetFromFolder, self).__init__()
        self.a_path = image_dir
        self.b_path = label_dir
        self.image_filenames = [x for x in listdir(self.a_path) if is_image_file(x)]
        self.augment = augment

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.augmentations = A.Compose([
            A.Resize(512, 512),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5)
        ])

    def __getitem__(self, index):
        a = cv2.imread(join(self.a_path, self.image_filenames[index]))
        b = cv2.imread(join(self.b_path, self.image_filenames[index]), cv2.IMREAD_GRAYSCALE)

        if a is None or b is None:
            raise RuntimeError(f"Failed to load image or label at index {index}")

        a = cv2.resize(a, (512, 512), interpolation=cv2.INTER_CUBIC)
        b = cv2.resize(b, (512, 512), interpolation=cv2.INTER_NEAREST)

        b = cv2.cvtColor(b, cv2.COLOR_GRAY2RGB)

        if self.augment:
            augmented = self.augmentations(image=a, mask=b)
            a = augmented['image']
            b = augmented['mask']

        a = Image.fromarray(a)
        b = Image.fromarray(b)
        a = self.transform(a)
        b = self.transform(b)

        return b, a

    def __len__(self):
        return len(self.image_filenames)


def get_training_set(image_dir, label_dir, augment=False):
    return DatasetFromFolder(image_dir, label_dir, augment=augment)


def get_test_set(image_dir, label_dir):
    return DatasetFromFolder(image_dir, label_dir, augment=False)
