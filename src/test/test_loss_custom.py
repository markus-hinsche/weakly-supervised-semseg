import numpy as np
import torch
from fastai.layers import CrossEntropyFlat

from ..loss_custom import WeakCrossEntropy


def test_cross_entropy():
    n_classes = 5
    bs = 2
    width = 200
    height = 200
    predictions = torch.rand(bs, n_classes, width, height, dtype=torch.float32)  # same shape as images
    ys = torch.empty([bs, 1, width, height]).random_(n_classes)

    loss = CrossEntropyFlat(axis=1)
    value = loss(predictions, ys.long())
    assert value>0

def test_weak_cross_entropy():
    n_classes = 5
    bs = 2
    width = 200
    height = 200
    predictions = torch.rand(bs, n_classes, width, height, dtype=torch.float32)  # same shape as images
    ys = ['11001', '01001']

    loss = WeakCrossEntropy(axis=1)
    value = loss(predictions, ys)
    assert value>0

def test_weak_cross_entropy_all_classes():
    n_classes = 5
    bs = 2
    width = 200
    height = 200
    predictions = torch.rand(bs, n_classes, width, height, dtype=torch.float32)  # same shape as images
    ys = ['11111', '11111']

    loss = WeakCrossEntropy(axis=1)
    value = loss(predictions, ys)
    # assert value==0.  # TODO

def test_weak_cross_entropy_one_color_correct():
    """When always predicting the correct color with prob=1.0, the loss should be zero"""
    n_classes = 5
    bs = 2
    width = 10
    height = 10

    # always predict color WHITE
    predictions = torch.zeros(bs, n_classes, width, height, dtype=torch.float32)  # same shape as images
    predictions[:, 0, :, :] = 1

    ys = ['10000', '10000']

    loss = WeakCrossEntropy(axis=1)
    value = loss(predictions, ys)
    assert value==0.

def test_weak_cross_entropy_one_color_wrong():
    """When always predicting the wrong color with prob=1.0, the loss should be zero"""
    n_classes = 5
    bs = 2
    width = 10
    height = 10

    # always predict color WHITE
    predictions = torch.zeros(bs, n_classes, width, height, dtype=torch.float32)  # same shape as images
    predictions[:, 1, :, :] = 1

    ys = ['00010', '00010']

    loss = WeakCrossEntropy(axis=1)
    value = loss(predictions, ys)
    assert value==np.inf
