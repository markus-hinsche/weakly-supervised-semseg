from typing import Tuple, List

import numpy as np
import torch


THRESH_LOWER_CLIP_PROBS = 0.001

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


class WeakCrossEntropy():
    """Cross Entropy for semantically segmented images based on weak labels.

    Weak labels are vector of color flags.
    Each flags represents whether a certain color exists in the image or not.
    """
    def __init__(self, codes, axis=1):
        self.axis = axis
        assert axis == 1
        self.codes = codes
        self.one_tensor = torch.Tensor([1.]).to(device)

    def __call__(self, input_orig, target):
        """
        input_orig: predictions of shape (bs,ncolors,width,height)

        target: of shape (bs, ncolors)
                Example tensor([[1., 1., 1., 1., 1.],
                                [1., 1., 1., 1., 0.]])
        """
        bs, ncolors, width, height = input_orig.shape
        assert target.shape[0] == bs

        ncolors_in_target = target.shape[1]
        if ncolors_in_target != ncolors:
            target_to_concat = torch.tensor(bs, ncolors - ncolors_in_target)
            target = torch.cat((target, target_to_concat), dim=1)

        input = input_orig.softmax(dim=1)

        # Flatten the input
        input = input.reshape(bs, ncolors, -1)  # shape (bs, ncolors, width*height)

        if torch.isnan(input).any():
            print("input is nan: ", input)

        target_mask = target.repeat(width*height, 1, 1).transpose(0, 1).transpose(1,2)  # shape (bs, ncolors, width*height)

        sums_prob_y_1 = (input*target_mask).sum(axis=1)  # shape (bs, width*height)

        # Clip low probabilities
        sums_prob_y_1[sums_prob_y_1 < THRESH_LOWER_CLIP_PROBS] = THRESH_LOWER_CLIP_PROBS

        item_losses = sums_prob_y_1.log() * -1.0  # shape (bs, width*height)

        assert item_losses.shape == (bs, width*height)
        assert not torch.isinf(item_losses).any()

        loss = item_losses.mean()

        if torch.isnan(loss).any():
            print("loss is nan, ", sums_prob_y_1)

        return loss
