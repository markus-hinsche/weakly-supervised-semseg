from typing import Tuple, List

import numpy as np
import torch


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


class WeakCrossEntropy():
    def __init__(self, codes, axis=1):
        self.axis = axis
        assert axis == 1
        self.codes = codes
        self.one_tensor = torch.Tensor([1.]).to(device)

    def __call__(self, input, target):
        """
        input  # shape(bs,ncolors,width,height)

        target = tensor([[1., 1., 1., 1., 1.],
                         [1., 1., 1., 1., 0.]])

        """
        assert len(input.shape) == 4
        bs, ncolors, width, height = input.shape
        assert target.shape == (bs, ncolors)

        # assert: prediction have gone through softmax
        input = input.softmax(dim=1)
        assert torch.isclose(input.sum(dim=1), self.one_tensor).all(), input.sum(dim=1)

        # flatten the input
        input = input.reshape(bs, ncolors, -1)  # shape(bs, ncolors, width*height)

        target_mask = target.repeat(width*height, 1, 1).transpose(0, 1).transpose(1,2)

        sums_prob_y_1 = (input*target_mask).sum(axis=1)  # shape (bs, width*height, )
        item_losses = sums_prob_y_1.log() * -1.0  # shape (bs, width*height, )

        assert item_losses.shape == (bs, width*height)
        return item_losses.mean()
