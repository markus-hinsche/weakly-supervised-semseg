from typing import List

import torch

from src.config import LABELS, RED, BLACK

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

name2id = {v:k for k,v in enumerate(LABELS+[RED, BLACK])}  # {WHITE:0, BLUE:1}
void_codes_red = name2id[RED]
void_codes_black = name2id[BLACK]


def acc_satellite(input: torch.tensor, target: torch.tensor) -> torch.tensor:
    """Calculate accuracy for semantic segmentation

    Args:
        input: Predictions
        target: Mask with label per pixel

    Returns:
        One value for the mean accuracy
    """
    target = target.squeeze(1)
    mask = target != void_codes_red
    mask = target != void_codes_black
    return (input.argmax(dim=1)[mask]==target[mask]).float().mean()


def acc_weakly(input: torch.tensor, target: List[int]) -> torch.tensor:
    """Calculate a metric based on the tile-level labels and per-pixel predictions

    Args:
        input: Predictions
        target: 0 or 1 for each of the 5 colors

    Returns:
        One value for the mean accuracy
    """
    bs, ncolors, width, height = input.shape
    input_ = input.reshape(bs, ncolors, -1)

    num_colors = 5
    if ncolors > num_colors:
        input_ = input_[:, 0:num_colors,:]
        ncolors = num_colors

    input_ = input_.argmax(dim=1)  # shape: (bs, pixel)

    mat = torch.zeros(input_.shape).to(device)
    for batch_idx in range(bs):
        for i in range(ncolors):
            if target[batch_idx][i] == 0:
                continue
            mat[batch_idx][input_[batch_idx]==i] = 1
    return mat.float().mean()
