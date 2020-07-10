import torch

from ..metrics_custom import acc_weakly
from ..config import CODES


def test_acc_weakly():
    n_classes = 5
    bs = 2
    width = 3
    height = 3
    predictions = torch.zeros(bs, n_classes, width, height, dtype=torch.float32)

    # Set color=3 for all pixels
    predictions[:, 3, :, :] = 0.5

    # Set color=1 for pixels of a specific width
    predictions[:, 1, 1, :] = 0.6

    # print(predictions.argmax(dim=1).reshape(bs, -1))

    ys = torch.tensor([[0,1,0,1,0], [0,1,0,0,0]])
    accuracy = acc_weakly(predictions, ys)

    assert torch.isclose(accuracy, torch.Tensor([2/3]))
