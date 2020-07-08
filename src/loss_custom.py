from typing import Tuple, List

import numpy as np
import torch


def get_colors_for_image(label_vector: str) -> Tuple[np.array]:
    """
    label_vector:
        '11001'
    output:
        colors_y_1 = [0,1,4]
        colors_y_0 = [2,3]
    """
    label_vector_arr = np.array(list(map(int,label_vector)))  # array([1, 1, 0, 0, 1])

    colors_y_1 = np.where(label_vector_arr == 1)[0]
    colors_y_0 = np.where(label_vector_arr == 0)[0]
    return colors_y_1, colors_y_0  # TODO speed-optimization: return only one part of the tuple


class WeakCrossEntropy():
    """Assumes that predictions have gone through a SoftmaxLayer"""
    def __init__(self, axis=1):
        self.axis = axis
        assert axis == 1

    def __call__(self, input, target):
        """
        input  # shape(bs,ncolors,width,height)
        target = ['11001', '00011', ...]
        """

        assert len(input.shape) == 4
#         assert len(target.shape) == 1  # vector of categories
        bs, ncolors, width, height = input.shape
        assert len(target) == bs, (len(target), bs)

        # assert: prediction have gone through softmax
        assert torch.isclose(input.sum(dim=1), torch.Tensor([1.])).all()

        # flatten the input
        input = input.reshape(bs, ncolors, -1)  # shape(bs, ncolors, width*height)

        # multi indexing ?
        # Problem: multiple colors_y_1 lists have different shapes
        # For now, have a little for loop running over samples in the batch

        sums_prob_y_1 = torch.empty(bs, width*height)
        sums_prob_y_0 = torch.empty(bs, width*height)

        for batch_idx in range(bs):
            # Get target indexes
            colors_y_1, colors_y_0 = get_colors_for_image(target[batch_idx])

            sums_prob_y_1[batch_idx] = input[batch_idx][colors_y_1].sum(axis=0)
            sums_prob_y_0[batch_idx] = input[batch_idx][colors_y_0].sum(axis=0)

        assert torch.isclose(sums_prob_y_1 + sums_prob_y_0, torch.Tensor([1.])).all()

        item_losses = sums_prob_y_1.log() * -1.0  # shape (bs, width*height, )
        # (1 - sum_prob_y_0).log() * -1.0  # same as item_losses due to assert

        assert item_losses.shape == (bs, width*height)
        return item_losses.mean()
