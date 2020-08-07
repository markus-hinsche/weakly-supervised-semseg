# https://docs.fast.ai/dev/test.html#getting-reproducible-results

from functools import partial
from pathlib import Path
import re

import torch

from config import (IMAGE_DATA_DIR, GT_DIR, IMAGE_DATA_TILES_DIR, GT_TILES_DIR,
                    GT_ADJ_TILES_DIR, TILES_DIR,
                    LABELS, RED, BLACK, N1, N2, N_validation, MODEL_DIR,
                    BASE_DIR
                   )


def set_seed(seed=42):
    # python RNG
    import random
    random.seed(seed)

    # pytorch RNGs
    import torch
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

    # numpy RNG
    import numpy as np
    np.random.seed(seed)

IMG_FILE_PREFIX = "top_mosaic_09cm_area"
REGEX_IMG_FILE_NAME = re.compile(fr"{IMG_FILE_PREFIX}(?P<area_id>\d+)_tile(?P<tile_id>\d+).tif")
REGEX_IMG_FILE_NAME_WITH_LABEL_VECTOR = re.compile(IMG_FILE_PREFIX + r"(?P<area_id>\d+)_tile(?P<tile_id>\d+)_(?P<label_vector>\d{5}).tif")

def _is_in_set(x, N, regex_obj):
    fname = x.name  # e.g.: top_mosaic_09cm_area30_tile120.tif'

    match_result = regex_obj.search(fname)
    area_id = match_result.group('area_id')
    tile_id = match_result.group('tile_id')
    image_fname = f"{IMG_FILE_PREFIX}{area_id}.tif"  # e.g.: top_mosaic_09cm_area30.tif'
    return image_fname in N


is_in_set_n1 = partial(_is_in_set, N=N1)
is_in_set_n2 = partial(_is_in_set, N=N2)
is_in_set_nvalidation = partial(_is_in_set, N=N_validation)
is_in_set_n1_or_nvalidation = partial(_is_in_set, N=N1+N_validation)
is_in_set_n2_or_nvalidation = partial(_is_in_set, N=N2+N_validation)


def get_y_fn(x):
    return BASE_DIR / GT_ADJ_TILES_DIR / x.name

def get_y_colors(x):
    fname = x.name
    match_result = REGEX_IMG_FILE_NAME_WITH_LABEL_VECTOR.search(fname)
    label_vector = match_result.group('label_vector')
    label_vector_arr = torch.tensor(list(map(int,label_vector))) # NEW

    indexes = torch.where(label_vector_arr == 1)[0]
    colors = [LABELS[idx] for idx in indexes]

    assert 0<len(indexes)<6, (len(indexes), x)

    return colors

def has_a_valid_color(x):
    fname = x.name
    match_result = REGEX_IMG_FILE_NAME_WITH_LABEL_VECTOR.search(fname)
    label_vector = match_result.group('label_vector')
    label_vector_arr = torch.tensor(list(map(int,label_vector))) # NEW

    indexes = torch.where(label_vector_arr == 1)[0]
    if not (0<len(indexes)<6):
        return False
    return True
