from typing import Tuple, List

import numpy as np
import torch


def get_colors_for_image(label_vector: str) -> Tuple[List[int]]:
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
    return list(colors_y_1), list(colors_y_0)


class WeakCrossEntropy():
    """Assumes that predictions have gone through a SoftmaxLayer"""
    def __init__(self, axis=1):
        self.axis = axis
        assert axis == 1

    def __call__(self, input, target):
        """
        input  # shape(ncolors,width,height)
        target = '11001'
        """

        assert len(input.shape) == 4
#         assert len(target.shape) == 1  # vector of categories
        bs, ncolors, width, height = input.shape
        assert len(target) == bs, (len(target), bs)

        # assert: prediction have gone through softmax
        assert torch.isclose(input.sum(dim=1), torch.Tensor([1.])).all()

        # flatten the input
        input = input.reshape(bs, ncolors, -1)  # shape(bs, ncolors, width*height)

        # Get target indexes
        colors_y_1, colors_y_0 = get_colors_for_image(target[0])  # TODO do for bs
        sum_prob_y_1 = input[0][colors_y_1].sum(axis=0)
        sum_prob_y_0 = input[0][colors_y_0].sum(axis=0)

        assert torch.isclose(sum_prob_y_1 + sum_prob_y_0, torch.Tensor([1.])).all()

        item_losses = sum_prob_y_1.log() * -1.0
        # (1 - sum_prob_y_0).log() * -1.0  # same as item_losses due to assert

        assert item_losses.shape[0] == width*height  # * bs ?
        return item_losses.mean()
