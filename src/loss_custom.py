from typing import Tuple, List

import torch


def get_target_indexes(label_vector: str) -> Tuple[List[int]]:
    """
    label_vector:
        '11001'
    output:
        colors_y_1 = [0,1,4]
        colors_y_0 = [2,3]
    """
    colors_y_1 = []
    colors_y_0 = []
    for i, character in enumerate(label_vector): # TODO remove loop, use masks
        if character == '1':
            colors_y_1.append(i)
        elif character == '0':
            colors_y_0.append(i)
        else:
            raise Exception("Unknown char:", character)
    return colors_y_1, colors_y_0


class WeakCrossEntropy():
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

        # flatten the input
        input = input.reshape(bs, ncolors, -1)  # shape(bs, ncolors, width*height)

        # Get target indexes
        colors_y_1, colors_y_0 = get_target_indexes(target[0])  # TODO do for bs
        yHat_y_1 = input[0][colors_y_1].log() * -1.0
        yHat_y_0 = (1 - input[0][colors_y_0]).log() * -1.0

        item_losses = torch.cat([yHat_y_1, yHat_y_0], dim=0).sum(dim=0)

        assert item_losses.shape[0] == width*height  # * bs ?
        return item_losses.mean()
