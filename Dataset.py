import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import ImageProcessing
from pathlib import Path

__all__ = [
    "ImageDataset",
    "train_collate_func",
    "val_test_collate_func"
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
    def __init__(self, dataset, labels_dict: dict = None,
                 image_width: int = None, image_height: int = None, mean: list = None,
                 std: list = None, mode: str = "train"):
        self.dataset = dataset
        self.labels_dict = labels_dict
        self.image_width = image_width
        self.image_height = image_height
        self.mean = mean
        self.std = std
        self.mode = mode

        self.images = self._load_data()

        if self.mode == "train" and self.labels_dict is None:
            raise ValueError("Labels dictionary required for training mode")

    def _load_data(self):
        images = self.dataset['image']
        labels = self.dataset['label']
        return images, labels

    def __getitem__(self, index: int):
        images = self.images

        # Read image in grayscale and resize (combine steps)
        image = cv2.cvtColor(images, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (self.image_width, self.image_height), interpolation=cv2.INTER_CUBIC)
        image = np.reshape(image, (self.image_height, self.image_width, 1))

        # Normalize and convert to Tensor format (potentially reuse function)
        image = ImageProcessing.img2tensor(image, mean=self.mean, std=self.std)

        if self.mode == "train":
            target = self.dataset['label']
            target_length = len(target)
            return image, target, target_length
        elif self.mode == "valid" or self.mode == "test":
            return image, self.dataset['label']
        else:
            raise ValueError("Unsupported data processing model, please use `train`, `valid` or `test`.")

