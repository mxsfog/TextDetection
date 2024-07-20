# Text Detection Repository

This repository implements a CRNN-based text detection system using the MJSynth dataset. The project consists of various components, each handling different parts of the text detection pipeline. Below is a description of each file and its role in the project.

## Files and Their Descriptions

### `conf.py`

This file contains the configuration settings for the text detection system. It sets up the random seed for reproducibility, defines the device for computation (CPU or GPU), and loads the dataset. It also specifies the model's hyperparameters and settings for training and testing modes.

- **Random Seeds**: Ensures reproducibility.
- **Device Configuration**: Chooses between CPU and GPU.
- **Dataset Loading**: Loads training and testing datasets.
- **Mode Configuration**: Sets parameters for training and testing.

### `dataset.py`

Defines the `ImageDataset` class used for loading and processing images. This class implements custom dataset handling for the text detection system.

- **`ImageDataset` Class**: Loads and preprocesses images from the dataset, including resizing, normalization, and conversion to tensors.
- **Collate Functions**: Defines how to batch data for training and validation/testing.

### `decoder.py`

Contains functions for decoding the model's output. It includes methods for reconstructing the text from the model's predictions and performing greedy decoding.

- **`reconstruct` Function**: Removes consecutive duplicate labels and blank labels.
- **`greedy_decode` Function**: Decodes the model's output probabilities into text labels.
- **`ctc_decoder` Function**: Uses the CTC (Connectionist Temporal Classification) loss to decode sequences of predictions into text.

### `imageProcessing.py`

Includes utility functions for image processing, specifically converting images into tensors.

- **`img2tensor` Function**: Converts a numpy array image into a PyTorch tensor, applies normalization, and optionally converts to half precision.

### `model.py`

Defines the CRNN model architecture used for text detection. It includes the convolutional layers and BiLSTM layers for feature extraction and sequence modeling.

- **`BiLSTM` Class**: Implements a bidirectional LSTM for sequence modeling.
- **`CRNN` Class**: Combines convolutional layers with BiLSTM for text recognition. Initializes weights for convolutional layers.

### `train_model.py`

Contains the main training loop and utility functions for training and validating the CRNN model.

- **`main` Function**: Starts the training process.
- **`load_dataset` Function**: Prepares the data loaders for training and testing.
- **`build_model` Function**: Constructs the CRNN model and moves it to the appropriate device.
- **`define_loss` Function**: Defines the loss function (CTC Loss).
- **`define_optimizer` Function**: Sets up the optimizer (Adadelta).
- **`train` Function**: Trains the model and saves it.
- **`val` Function**: Validates the model and computes accuracy.

### Requirements

- Python 3.x
- PyTorch
- OpenCV
- Numpy
- Hugging Face Datasets

### Installation

1. Clone the repository:
    ```bash
    git clone <repository_url>
    ```
2. Install the required packages:
    ```bash
    pip install torch opencv-python-headless numpy datasets
    ```

### Usage

1. **Training**:
    ```bash
    python train_model.py
    ```

2. **Testing/Validation**:
    The `train_model.py` script handles both training and validation. Adjust the configuration in `conf.py` for the desired mode.
