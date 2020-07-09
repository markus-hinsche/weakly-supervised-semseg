from typing import Tuple, List

import numpy as np
import torch


HIGH_LOSS = 7.

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


class WeakCrossEntropy():
    def __init__(self, codes, axis=1):
        self.axis = axis
        assert axis == 1
        self.codes = codes
        self.one_tensor = torch.Tensor([1.]).to(device)

    def __call__(self, input_orig, target):
        """
        input: predictions # shape(bs,ncolors,width,height)

        target = tensor([[1., 1., 1., 1., 1.],
                         [1., 1., 1., 1., 0.]])

        """
        assert len(input_orig.shape) == 4
        bs, ncolors, width, height = input_orig.shape
        assert target.shape == (bs, ncolors)

        input = input_orig.log_softmax(dim=1)

        # flatten the input
        input = input.reshape(bs, ncolors, -1)  # shape(bs, ncolors, width*height)

        target_mask = target.repeat(width*height, 1, 1).transpose(0, 1).transpose(1,2)

        sums_prob_y_1 = (input*target_mask).sum(axis=1)  # shape (bs, width*height, )
        item_losses = sums_prob_y_1 * -1.0  # shape (bs, width*height, )

        assert item_losses.shape == (bs, width*height)

        # Clip loss
        item_losses[torch.isinf(item_losses)] = HIGH_LOSS

        loss = item_losses.mean()

        return loss
