import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import ImageProcessing
from pathlib import Path

__all__ = [

]


def train_collate_func(batch):
    images, target, target_len = zip(*batch)
    images = torch.stack(images, 0)
    target = torch.cat(target, 0)
    target_len = torch.cat(target_len, 0)

    return images, target, target_len

def val_test_collate_func(batch):
    img_path, images, target = zip(*batch)
    images = torch.stack(images, 0)

    return img_path, images, target



class ImageDataset(Dataset):
    def __init__(self, dataroot: str, annotation_file_name: str, labels_dict: dict = None,
                 image_width: int = None, image_height: int = None, mean: list = None,
                 std: list = None, mode: str = "train"):
        self.dataroot = Path(dataroot)  # Use Pathlib for robust path handling
        self.annotation_file_name = self.dataroot / annotation_file_name  # Combine paths
        self.labels_dict = labels_dict
        self.image_width = image_width
        self.image_height = image_height
        self.mean = mean
        self.std = std
        self.mode = mode

        # Load image paths and targets directly
        self.images_path, self.images_target = self._load_data()

        if self.mode == "train" and self.labels_dict is None:
            raise ValueError("Labels dictionary required for training mode")

    def _load_data(self):
        images_path = []
        images_target = []
        with open(self.annotation_file_name, "r", encoding="UTF-8") as f:
            for line in f.readlines():
                image_path, image_target = line.strip().split(" ")
                images_path.append(self.dataroot / image_path)  # Combine paths
                images_target.append(image_target)
        return images_path, images_target

    def __getitem__(self, index: int) -> [str, torch.Tensor, torch.Tensor, torch.Tensor]:
        image_path = self.images_path[index]

        # Read image in grayscale and resize (combine steps)
        image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (self.image_width, self.image_height), interpolation=cv2.INTER_CUBIC)
        image = np.reshape(image, (self.image_height, self.image_width, 1))

        # Normalize and convert to Tensor format (potentially reuse function)
        image = ImageProcessing.img2tensor(image, mean=self.mean, std=self.std)

        if self.mode == "train":
            target = [self.labels_dict[character] for character in self.images_target[index]]
            target = torch.LongTensor(target)
            target_length = torch.LongTensor([len(target)])
            return image, target, target_length
        elif self.mode == "valid" or self.mode == "test":
            return image_path, image, self.images_target[index]
        else:
            raise ValueError("Unsupported data processing model, please use `train`, `valid` or `test`.")

    def __len__(self):
        return len(self.images_path)
