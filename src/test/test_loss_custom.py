from typing import List

import numpy as np
import torch
from torch import nn
from fastai.layers import CrossEntropyFlat

from ..loss_custom import WeakCrossEntropy


def test_cross_entropy():
    n_classes = 5
    bs = 2
    width = 200
    height = 200

    predictions = torch.rand(
        bs, n_classes, width, height, dtype=torch.float32
    )  # same shape as images
    # apply Softmax along the ncolors dimension
    predictions = nn.Softmax(dim=1)(predictions)

    ys = torch.empty([bs, 1, width, height]).random_(n_classes)

    loss = CrossEntropyFlat(axis=1)
    value = loss(predictions, ys.long())
    assert value > 0


def test_weak_cross_entropy_basic():
    n_classes = 5
    bs = 2
    width = 200
    height = 200

    predictions = torch.rand(
        bs, n_classes, width, height, dtype=torch.float32
    )  # same shape as images
    # apply Softmax along the ncolors dimension
    predictions = nn.Softmax(dim=1)(predictions)

    ys = torch.tensor([[1, 1, 0, 0, 1], [0, 1, 0, 0, 1]])

    loss = WeakCrossEntropy()
    value = loss(predictions, ys)
    assert value > 0


def test_weak_cross_entropy_all_classes():
    n_classes = 5
    bs = 2
    width = 200
    height = 200

    predictions = torch.rand(
        bs, n_classes, width, height, dtype=torch.float32
    )  # same shape as images
    # apply Softmax along the ncolors dimension
    predictions = nn.Softmax(dim=1)(predictions)

    ys = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]])

    loss = WeakCrossEntropy()
    value = loss(predictions, ys)
    assert torch.isclose(value, torch.Tensor([0.0]))


def test_weak_cross_entropy_one_color_correct():
    """When always predicting the correct color with prob=1.0, the loss should be zero"""
    n_classes = 5
    bs = 2
    width = 10
    height = 10

    # always predict a specific color
    predictions = torch.zeros(
        bs, n_classes, width, height, dtype=torch.float32
    )  # same shape as images
    predictions[:, 3, :, :] = 1

    ys = torch.tensor([[0, 0, 0, 1, 0], [1, 0, 0, 1, 0]])

    loss = WeakCrossEntropy()
    value = loss(predictions, ys)

    # Assert small loss
    assert value < 0.8


def test_weak_cross_entropy_one_color_wrong():
    """When always predicting the wrong color with prob=1.0, the loss should be zero"""
    n_classes = 5
    bs = 2
    width = 10
    height = 10

    # always predict a specific color
    predictions = torch.zeros(
        bs, n_classes, width, height, dtype=torch.float32
    )  # same shape as images
    predictions[:, 1, :, :] = 1

    ys = torch.tensor([[0, 0, 0, 1, 0], [0, 0, 0, 1, 0]])

    loss = WeakCrossEntropy()
    value = loss(predictions, ys)

    # Assert big loss
    assert value > 1.8
