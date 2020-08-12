import torch

from ..metrics_custom import acc_satellite, acc_weakly


def test_acc_satellite():
    n_classes = 5
    bs = 2
    width = 3
    height = 3

    predictions = torch.zeros(bs, n_classes, width, height, dtype=torch.float32)
    # Set color=3 for all pixels
    predictions[:, 3, :, :] = 0.5
    # Set color=1 for pixels of a specific width
    predictions[:, 1, 1, :] = 0.6

    # Set color=3 for all pixels
    ys = torch.ones(bs, 1, width, height, dtype=torch.float32)*3
    # Set color=1 for pixels of a specific width
    ys[:, 0, 1, :] = 1

    accuracy = acc_satellite(predictions, ys)
    assert torch.isclose(accuracy, torch.Tensor([1.]))

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

    ys = torch.tensor([[0,1,0,1,0], [0,1,0,0,0]])
    accuracy = acc_weakly(predictions, ys)

    assert torch.isclose(accuracy, torch.Tensor([2/3]))

def test_acc_weakly_with_7color_prediction():
    n_classes = 7
    bs = 2
    width = 3
    height = 3
    predictions = torch.zeros(bs, n_classes, width, height, dtype=torch.float32)

    predictions[:, 3, :, :] = 0.5
    predictions[:, 1, 1, :] = 0.6

    ys = torch.tensor([[0,1,0,1,0], [0,1,0,0,0]])
    accuracy = acc_weakly(predictions, ys)

    assert torch.isclose(accuracy, torch.Tensor([2/3]))
