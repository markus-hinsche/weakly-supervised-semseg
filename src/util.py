# https://docs.fast.ai/dev/test.html#getting-reproducible-results

from functools import partial
import re

from config import (IMAGE_DATA_DIR, GT_DIR, IMAGE_DATA_TILES_DIR, GT_TILES_DIR,
                    GT_ADJ_TILES_DIR, TILES_DIR,
                    LABELS, RED, BLACK, N1, N2, N_validation, MODEL_DIR
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
