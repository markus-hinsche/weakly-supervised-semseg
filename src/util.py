# https://docs.fast.ai/dev/test.html#getting-reproducible-results

from functools import partial
from pathlib import Path
import re
from typing import List, Tuple

import torch
from fastai.vision.image import ImageSegment
from fastai.basic_train import Learner

from src.constants import (
    GT_ADJ_TILES_DIR,
    CLASSES,
    ALL_CLASSES,
    N1,
    N2,
    N_validation,
    BASE_DIR,
)


def set_seed(seed: int = 42):
    # python RNG (random number generator)
    import random

    random.seed(seed)

    # pytorch RNGs
    import torch

    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # numpy RNG
    import numpy as np

    np.random.seed(seed)


IMG_FILE_PREFIX = "top_mosaic_09cm_area"
REGEX_IMG_FILE_NAME = re.compile(
    fr"{IMG_FILE_PREFIX}(?P<area_id>\d+)_tile(?P<tile_id>\d+).tif"
)
REGEX_IMG_FILE_NAME_WITH_LABEL_VECTOR = re.compile(
    IMG_FILE_PREFIX
    + r"(?P<area_id>\d+)_tile(?P<tile_id>\d+)_(?P<label_vector>\d{5}).tif"
)


def _is_in_imageset(x: Path, images: List[str], regex_obj: re.Pattern) -> bool:
    """Determine if in a sample is part of a list of images.

    Args:
        x: Input posix path
        images: The set of images
        Example: ['top_mosaic_09cm_area38.tif',
                  'top_mosaic_09cm_area24.tif']
        regex_obj (re.Pattern): A compiled regex

    Returns:
        bool: If in imageset or not
    """
    fname = x.name  # e.g.: top_mosaic_09cm_area30_tile120.tif

    match_result = regex_obj.search(fname)
    area_id = match_result.group("area_id")
    image_fname = f"{IMG_FILE_PREFIX}{area_id}.tif"  # e.g.: top_mosaic_09cm_area30.tif
    return image_fname in images


is_in_set_n1 = partial(_is_in_imageset, images=N1)
is_in_set_n2 = partial(_is_in_imageset, images=N2)
is_in_set_nvalidation = partial(_is_in_imageset, images=N_validation)
is_in_set_n1_or_nvalidation = partial(_is_in_imageset, images=N1 + N_validation)
is_in_set_n2_or_nvalidation = partial(_is_in_imageset, images=N2 + N_validation)


def get_y_fn(x: Path) -> str:
    return BASE_DIR / GT_ADJ_TILES_DIR / x.name


def get_y_colors(x: Path) -> List[Tuple[int, int, int]]:
    """Read the filename which contains the label vector

    The label vector is a 5 digit number (e.g. 11001) describing which
    colors are contained in the entire image

    Args:
        x: Posix path of the tile's image file

    Returns:
        colors
        Example: [(255, 255, 255), (0, 0, 255), (255, 255, 0)]
    """
    fname = x.name
    match_result = REGEX_IMG_FILE_NAME_WITH_LABEL_VECTOR.search(fname)
    label_vector = match_result.group("label_vector")
    label_vector_arr = torch.tensor(list(map(int, label_vector)))

    indexes = torch.where(label_vector_arr == 1)[0]
    assert 0 < len(indexes) <= len(CLASSES), (len(indexes), x)

    colors = [CLASSES[idx] for idx in indexes]
    return colors


def has_a_valid_color(x: Path) -> bool:
    fname = x.name
    match_result = REGEX_IMG_FILE_NAME_WITH_LABEL_VECTOR.search(fname)
    label_vector = match_result.group("label_vector")
    label_vector_arr = torch.tensor(list(map(int, label_vector)))

    indexes = torch.where(label_vector_arr == 1)[0]

    # There must be at least one color and at maximum num_classes colors
    return 0 < len(indexes) <= len(CLASSES)


def show_prediction_vs_actual(sample_idx: int, learn: Learner) -> ImageSegment:
    """Return predicted mask, additionally print input image and tile-level label"""
    sample = learn.data.valid_ds[sample_idx]
    image, label = sample
    print("Label: " + str(label.__repr__()))
    image.show()
    batch = learn.data.one_item(image)
    pred = learn.pred_batch(batch=batch).squeeze(dim=0)
    img = pred.argmax(dim=0, keepdim=True)

    predicted_colors = torch.zeros(len(ALL_CLASSES))
    for i in img.unique():
        predicted_colors[i] = 1
    print("Predicted colors: " + str(predicted_colors))

    image_segment = ImageSegment(img)
    return image_segment
