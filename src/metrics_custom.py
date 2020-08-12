from typing import List

import torch

from src.constants import ALL_CLASSES, CLASSES, RED, BLACK

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NAME2ID = {v: k for k, v in enumerate(ALL_CLASSES)}
VOID_CODES_RED = NAME2ID[RED]
VOID_CODES_BLACK = NAME2ID[BLACK]


def acc_satellite(pred: torch.tensor, target: torch.tensor) -> torch.tensor:
    """Calculate accuracy for semantic segmentation

    Args:
        pred: Predictions
        target: Mask with label per pixel

    Returns:
        One value for the mean accuracy
    """
    target = target.squeeze(1)
    mask = target != VOID_CODES_RED
    mask = target != VOID_CODES_BLACK
    return (pred.argmax(dim=1)[mask] == target[mask]).float().mean()


def acc_weakly(pred: torch.tensor, target: List[int]) -> torch.tensor:
    """Calculate a metric based on the tile-level labels and per-pixel predictions

    Args:
        pred: Predictions
        target: List of numbers 0 or 1 for each of colors

    Returns:
        One value for the mean accuracy
    """
    bs, ncolors, _width, _height = pred.shape
    pred_ = pred.reshape(bs, ncolors, -1)

    # If the prediction is too big, reduce it to num_colors
    num_colors = len(CLASSES)
    if ncolors > num_colors:
        pred_ = pred_[:, 0:num_colors, :]
        ncolors = num_colors

    pred_ = pred_.argmax(dim=1)  # shape: (bs, pixel)

    # Set those pixels to one that appear in the tile-level label
    mask = torch.zeros(pred_.shape).to(DEVICE)
    for batch_idx in range(bs):
        for i in range(ncolors):
            if target[batch_idx][i] == 0:
                continue
            mask[batch_idx][pred_[batch_idx] == i] = 1

    return mask.float().mean()
