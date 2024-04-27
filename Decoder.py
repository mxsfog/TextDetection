import torch
import numpy as np

__all__ = [
    "decoder"
]

def reconstruct(labels, blank: int = 0) -> list:
    new_labels = []
    prev = None
    for label in labels:
        if label != prev and label != blank:  # Combine checks for non-blank and change
            new_labels.append(label)
        prev = label
    return new_labels

def greedy_decode(seq_log_probas, blank: int=0):
    labels = torch.argmax(seq_log_probas, dim=-1)
    return reconstruct(labels.cpu().numpy(), blank=blank)  # Process on CPU


def ctc_decode(log_probas: torch.Tensor, chars: dict, blank=0):
    seq_log_probas = np.transpose(log_probas.cpu().numpy(), (1, 0, 2))

    # Define decoded label and list of char names
    decoded_labels = []
    decoded_chars = []

    for seq_log_prob in seq_log_probas:
        decoded = greedy_decode(seq_log_prob, blank)
        decoded_char = [chars[key] for key in decoded]
        decoded_labels.append(decoded)
        decoded_chars.append(decoded_char)

    return decoded_labels, decoded_chars