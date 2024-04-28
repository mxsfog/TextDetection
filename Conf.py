import random
import numpy as np
import torch
from torch.backends import cudnn

random.seed(0)
torch.manual_seed(0)
np.random.seed(0)

device = "cuda" if torch.cuda.is_available() else "cpu"
cudnn.benchmark = True if device == "cuda" else False

chars = "0123456789abcdefghijklmnopqrstuvwxyz"
labels = {char: i+1 for i, char in enumerate(chars)}
chars_dict = {label: char for char, label in labels.items()}
model_num_classes = len(chars) + 1
model_image_width = 100
model_image_height = 32
mean = 0.5
std = 0.5
mode = 'train'

exp_name = f"CRNN_{mode.upper()}"  # Use uppercase for train/test mode
mode_config = {
    "train": {
        "train_dataroot": "./data/MJSynth",
        "annotation_train_file_name": "annotation_train.txt",
        "test_dataroot": "./data/IIIT5K",
        "annotation_test_file_name": "annotation_test.txt",
        "batch_size": 64,
        "num_workers": 4,
        "resume": "",
        "epochs": 5,
        "model_lr": 1.0,
        "print_frequency": 1000,
    },
    "test": {
        "fp16": True,
        "result_dir": "./results/test",
        "result_file_name": "IIIT5K_result.txt",
        "dataroot": "./data/IIIT5K",
        "annotation_file_name": "annotation_test.txt",
        "model_path": "results/pretrained_models/CRNN-MJSynth-e9341ede.pth.tar"
    }
}

# Load configuration based on mode
config = mode_config[mode]
