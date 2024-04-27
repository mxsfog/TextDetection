import cv2
import torch
import numpy as np

__all__ = [
    "img2tensor"
]


def img2tensor(image, range_norm=True, mean=0.5, std=0.5, half=False):

    tensor = torch.from_numpy(image.astype(np.float32) / 255.)

    # Permute dimensions and perform remaining operations
    tensor = tensor.permute(2, 0, 1)  # Move channel dimension to front
    if range_norm:
        tensor.sub_(mean).div_(std)
    if half:
        tensor = tensor.half()

    return tensor
